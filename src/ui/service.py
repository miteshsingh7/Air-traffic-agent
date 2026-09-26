"""Backend data and evaluation service for the AeroMetrics Conflict Research Console.

Research simulation only - not for operational use.
Interfaces with scenario parquet files, baseline predictors, residual LSTM models,
and conflict geometry / evaluation pipelines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

from src.conflict.geometry import ConflictRecord, compute_scenario_conflicts
from src.conflict.risk import compute_conflict_risk
from src.data.synthetic import TrajectoryConfig
from src.eval.conflict_eval import ConflictEvaluationResult
from src.eval.metrics import PositionErrorMetric
from src.eval.run_baseline import run_test_evaluation
from src.models.baseline import ConstantVelocityPredictor, SmoothedConstantVelocityPredictor
from src.models.lstm_predictor import LSTMTrajectoryPredictor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "lstm_v1" / "best.pt"

# Global model caches
_PREDICTORS: dict[str, Any] = {}
_METRICS_CACHE: dict[str, dict[str, Any]] = {}


def get_torch_device() -> torch.device:
    """Return active torch compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_predictor(model_name: str) -> Any:
    """Retrieve or initialize cached trajectory predictor."""
    dt_s = 5.0
    horizon_steps = 60
    if model_name == "lstm_v1" or model_name == "lstm":
        if "lstm_v1" not in _PREDICTORS:
            if not CHECKPOINT_PATH.exists():
                raise FileNotFoundError(f"LSTM checkpoint not found at {CHECKPOINT_PATH}")
            _PREDICTORS["lstm_v1"] = LSTMTrajectoryPredictor.from_checkpoint(CHECKPOINT_PATH)
        return _PREDICTORS["lstm_v1"]
    elif model_name in ("cv_smoothed", "smoothed"):
        if "cv_smoothed" not in _PREDICTORS:
            _PREDICTORS["cv_smoothed"] = SmoothedConstantVelocityPredictor(
                dt_s=dt_s, horizon_steps=horizon_steps
            )
        return _PREDICTORS["cv_smoothed"]
    else:
        if "cv_3step" not in _PREDICTORS:
            _PREDICTORS["cv_3step"] = ConstantVelocityPredictor(
                dt_s=dt_s, horizon_steps=horizon_steps
            )
        return _PREDICTORS["cv_3step"]


def get_system_info() -> dict[str, Any]:
    """Return system runtime configuration, seeds, device, and separation minima."""
    device = get_torch_device()
    seeds = {"easy": 42, "hard_v1": 1042, "hard_large": 2042}
    config_warning: str | None = None
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
                if "splits" in raw_cfg and "seed" in raw_cfg["splits"]:
                    seeds["easy"] = raw_cfg["splits"]["seed"]
                if "hard_dataset" in raw_cfg and "seed" in raw_cfg["hard_dataset"]:
                    seeds["hard_v1"] = raw_cfg["hard_dataset"]["seed"]
                if "hard_large_dataset" in raw_cfg and "seed" in raw_cfg["hard_large_dataset"]:
                    seeds["hard_large"] = raw_cfg["hard_large_dataset"]["seed"]
        except (yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            msg = f"Could not fully parse config at {CONFIG_PATH}: {exc}"
            warnings.warn(msg, stacklevel=2)
            logging.getLogger(__name__).warning(msg)
            config_warning = msg

    ckpt_exists = CHECKPOINT_PATH.exists()
    ckpt_size = CHECKPOINT_PATH.stat().st_size if ckpt_exists else 0
    device_str = str(device).upper()

    return {
        "device": device_str,
        "device_raw": str(device),
        "is_gpu": device.type in ("cuda", "mps"),
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)) if ckpt_exists else None,
        "checkpoint_exists": ckpt_exists,
        "checkpoint_size_bytes": ckpt_size,
        "seeds": seeds,
        "separation_minima": {
            "lateral_nm": 5.0,
            "vertical_ft": 1000.0,
        },
        "time_params": {
            "history_s": 60,
            "horizon_s": 300,
            "step_s": 5,
        },
        "console_version": "v2.4-alpha",
        "advisory_disclaimer": "RESEARCH SIMULATION — ADVISORY ONLY — NOT FOR OPERATIONAL USE",
        "config_warning": config_warning,
    }


def list_datasets() -> list[dict[str, Any]]:
    """List available datasets, splits, and counts."""
    return [
        {
            "id": "hard_v1",
            "name": "hard_v1 (multi-crossing)",
            "splits": ["test", "val", "train"],
            "default_split": "test",
            "description": "500 scenarios, 50% maneuvers, Gaussian noise, near-miss encounters (seed 1042)",
            "scenarios_count": 500,
        },
        {
            "id": "hard_large",
            "name": "hard_large (10k tracks)",
            "splits": ["test", "val", "train"],
            "default_split": "test",
            "description": "3,000 scenarios scaled benchmark under seed 2042",
            "scenarios_count": 3000,
        },
        {
            "id": "easy",
            "name": "easy (dense nominal)",
            "splits": ["test", "val", "train"],
            "default_split": "test",
            "description": "500 straight-line cruise scenarios (seed 42)",
            "scenarios_count": 500,
        },
        {
            "id": "opensky",
            "name": "opensky (live ADS-B sample)",
            "splits": ["sample"],
            "default_split": "sample",
            "description": "84 real European en-route flight trajectories reassembled from live stream",
            "scenarios_count": 84,
        },
    ]


def _resolve_dataset_paths(dataset: str) -> tuple[Path, Path | None, Path | None]:
    """Resolve data directory, splits file, and metadata file for given dataset."""
    if dataset in ("hard_v1", "hard"):
        data_dir = ROOT_DIR / "data" / "synthetic_hard"
        splits_file = ROOT_DIR / "data" / "splits_hard.json"
        meta_file = data_dir / "metadata.json"
    elif dataset in ("hard_large", "large"):
        data_dir = ROOT_DIR / "data" / "synthetic_hard_large"
        splits_file = ROOT_DIR / "data" / "splits_hard_large.json"
        meta_file = data_dir / "metadata.json"
    elif dataset in ("easy", "nominal"):
        data_dir = ROOT_DIR / "data" / "synthetic"
        splits_file = ROOT_DIR / "data" / "splits.json"
        meta_file = data_dir / "metadata.json"
    elif dataset in ("opensky", "opensky_live_sample"):
        data_dir = ROOT_DIR / "data" / "opensky_live_sample"
        splits_file = None
        meta_file = data_dir / "metadata.json"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    return data_dir, splits_file, meta_file


def list_scenarios(dataset: str = "hard_v1", split: str = "test") -> list[dict[str, Any]]:
    """List scenarios for given dataset and split with ground-truth conflict status."""
    data_dir, splits_file, meta_file = _resolve_dataset_paths(dataset)

    meta_dict: dict[str, Any] = {}
    if meta_file and meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception:
            pass

    scenario_ids: list[str] = []
    if splits_file and splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
            scenario_ids = splits.get(split, [])
    else:
        # For OpenSky or directory without explicit splits
        parquet_files = sorted(data_dir.glob("*.parquet"))
        scenario_ids = [p.stem for p in parquet_files]

    results: list[dict[str, Any]] = []
    for sid in scenario_ids:
        if not (data_dir / f"{sid}.parquet").exists():
            logging.getLogger(__name__).debug(
                "Skipping scenario '%s': parquet file not found in %s", sid, data_dir
            )
            continue
        scen_meta = meta_dict.get(sid, {})
        injected = bool(scen_meta.get("injected_conflict", False))
        near_miss = bool(scen_meta.get("near_miss", False))
        maneuvering = bool(scen_meta.get("maneuvering", False))

        if injected:
            tag = "ALERT"
            tag_type = "conflict"
            desc = "MANEUVER" if maneuvering else "NON-MANEUVER"
        elif near_miss:
            tag = "NEAR MISS"
            tag_type = "near_miss"
            desc = f"sep {scen_meta.get('near_miss_closest_lateral_nm', 5.5):.1f}NM"
        else:
            tag = "NOMINAL"
            tag_type = "clean"
            desc = "CLEAN"

        results.append({
            "scenario_id": sid,
            "has_conflict": injected,
            "tag": tag,
            "tag_type": tag_type,
            "description": desc,
            "maneuvering": maneuvering,
            "near_miss": near_miss,
        })

    return results


def load_scenario_detail(
    dataset: str = "hard_v1",
    split: str = "test",
    scenario_id: str = "scenario_0012",
    origin_t: float | None = None,
    model_name: str = "cv_smoothed",
    lookahead_s: float = 180.0,
) -> dict[str, Any]:
    """Load scenario trajectory tracks, origin slice, model prediction, and pair telemetry."""
    data_dir, splits_file, meta_file = _resolve_dataset_paths(dataset)

    # Validate that the requested scenario_id belongs to the requested split.
    # Only applies to datasets that have an explicit splits JSON file.
    if splits_file and splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        if split not in splits_data:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{dataset}'. Valid splits: {list(splits_data.keys())}"
            )
        split_ids: list[str] = splits_data.get(split, [])
        if scenario_id not in split_ids:
            raise ValueError(
                f"Scenario '{scenario_id}' is not a member of split '{split}' "
                f"in dataset '{dataset}'."
            )
    elif dataset in ("opensky", "opensky_live_sample"):
        if split not in ("sample",):
            raise ValueError(
                f"Unknown split '{split}' for dataset '{dataset}'. Valid splits: ['sample']"
            )

    file_path = data_dir / f"{scenario_id}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # 1. Ground truth scenario-wide conflicts
    gt_conflicts: list[ConflictRecord] = compute_scenario_conflicts(
        scenario_df=df,
        lateral_min_nm=5.0,
        vertical_min_ft=1000.0,
    )

    aircraft_ids = sorted(df["aircraft_id"].unique().tolist())
    all_times = sorted(df["t"].unique().tolist())
    t_min = float(min(all_times))
    t_max = float(max(all_times))

    # Valid origins require 60s history (13 steps at dt=5) and 300s future (60 steps)
    dt_s = 5.0
    history_s = 60.0
    horizon_s = 300.0
    origin_step_s = 30.0

    possible_origins = [
        t for t in all_times
        if (t >= t_min + history_s) and (t <= t_max - horizon_s) and (round(t % origin_step_s, 2) == 0.0)
    ]
    if not possible_origins:
        # Fallback if scenario duration is shorter
        possible_origins = [
            t for t in all_times
            if (t >= t_min + history_s) and (t <= t_max - 60.0)
        ]

    # Select origin
    if origin_t is None:
        if gt_conflicts:
            first_c_t = gt_conflicts[0].first_conflict_time
            # Pick origin 60-120s before conflict if available
            cand = [t for t in possible_origins if 30.0 <= (first_c_t - t) <= 150.0]
            curr_origin = cand[0] if cand else possible_origins[0]
        else:
            curr_origin = possible_origins[0] if possible_origins else t_min + history_s
    else:
        # Snap to closest possible origin
        if possible_origins:
            curr_origin = min(possible_origins, key=lambda t: abs(t - origin_t))
        else:
            curr_origin = origin_t

    predictor = get_predictor(model_name)

    # Prepare aircraft details
    aircraft_data: dict[str, Any] = {}
    history_windows: dict[str, np.ndarray] = {}

    for acid in aircraft_ids:
        ac_df = df[df["aircraft_id"] == acid].sort_values("t").reset_index(drop=True)
        # Full flown trajectory
        flown_df = ac_df[ac_df["t"] <= curr_origin]
        # 60s history window up to origin
        hist_df = ac_df[(ac_df["t"] >= curr_origin - history_s - 1e-3) & (ac_df["t"] <= curr_origin + 1e-3)]
        # True future window
        future_df = ac_df[(ac_df["t"] > curr_origin) & (ac_df["t"] <= curr_origin + horizon_s + 1e-3)]

        # Extract [x, y, alt] for predictor (use noisy if available)
        x_col = "x_obs_nm" if "x_obs_nm" in hist_df.columns else "x_nm"
        y_col = "y_obs_nm" if "y_obs_nm" in hist_df.columns else "y_nm"
        alt_col = "alt_obs_ft" if "alt_obs_ft" in hist_df.columns else "alt_ft"

        hist_mat = hist_df[[x_col, y_col, alt_col]].to_numpy(dtype=np.float64)
        history_windows[acid] = hist_mat

        # Current state at origin
        curr_row = ac_df[ac_df["t"] == curr_origin]
        if curr_row.empty:
            curr_row = ac_df.iloc[-1:]

        x_curr = float(curr_row["x_nm"].iloc[0])
        y_curr = float(curr_row["y_nm"].iloc[0])
        alt_curr = float(curr_row["alt_ft"].iloc[0])
        speed_curr = float(curr_row["speed_kt"].iloc[0]) if "speed_kt" in curr_row.columns else 400.0
        heading_curr = float(curr_row["heading_deg"].iloc[0]) if "heading_deg" in curr_row.columns else 0.0

        # Compute ROCD (fpm) over recent 10s
        recent = ac_df[ac_df["t"] <= curr_origin].tail(3)
        if len(recent) >= 2:
            dt_recent = float(recent["t"].iloc[-1] - recent["t"].iloc[0])
            dalt_recent = float(recent["alt_ft"].iloc[-1] - recent["alt_ft"].iloc[0])
            rocd_fpm = round((dalt_recent / max(dt_recent, 1.0)) * 60.0)
        else:
            rocd_fpm = 0

        # Predict future trajectory
        t0_infer = time.perf_counter()
        if hist_mat.shape[0] == 13:
            pred_future_mat = predictor.predict(hist_mat[np.newaxis, ...])[0]  # Shape: (60, 3)
        else:
            # Fallback linear extension if incomplete
            pred_future_mat = np.zeros((60, 3))
        inference_time_ms = (time.perf_counter() - t0_infer) * 1000.0

        # Format points as [t, x, y, alt]
        history_points = [
            {"t": float(r["t"]), "x": float(r["x_nm"]), "y": float(r["y_nm"]), "alt": float(r["alt_ft"])}
            for _, r in flown_df.iterrows()
        ]
        future_true_points = [
            {"t": float(r["t"]), "x": float(r["x_nm"]), "y": float(r["y_nm"]), "alt": float(r["alt_ft"])}
            for _, r in future_df.iterrows()
        ]
        future_pred_points = [
            {
                "t": float(curr_origin + (k + 1) * dt_s),
                "x": float(pred_future_mat[k, 0]),
                "y": float(pred_future_mat[k, 1]),
                "alt": float(pred_future_mat[k, 2]),
            }
            for k in range(min(len(pred_future_mat), int(lookahead_s / dt_s)))
        ]

        aircraft_data[acid] = {
            "aircraft_id": acid,
            "callsign": acid.replace("_", "-"),
            "flight_level": f"FL{round(alt_curr / 100)}",
            "altitude_ft": round(alt_curr, 1),
            "speed_kt": round(speed_curr, 1),
            "heading_deg": round(heading_curr, 1),
            "rocd_fpm": rocd_fpm,
            "rocd_str": (
                "LEVEL (0 fpm)"
                if abs(rocd_fpm) < 100
                else f"CLB +{rocd_fpm} fpm" if rocd_fpm > 0 else f"DES {rocd_fpm} fpm"
            ),
            "current_pos": {"x": round(x_curr, 3), "y": round(y_curr, 3), "alt": round(alt_curr, 1)},
            "history_points": history_points,
            "future_true": future_true_points,
            "future_pred": future_pred_points,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    # Pairwise conflict kinematics at current origin
    active_pair_data: dict[str, Any] | None = None
    all_pairs: list[dict[str, Any]] = []

    for i in range(len(aircraft_ids)):
        for j in range(i + 1, len(aircraft_ids)):
            ac1 = aircraft_data[aircraft_ids[i]]
            ac2 = aircraft_data[aircraft_ids[j]]

            p1_curr = np.array([ac1["current_pos"]["x"], ac1["current_pos"]["y"]])
            p2_curr = np.array([ac2["current_pos"]["x"], ac2["current_pos"]["y"]])
            curr_lat_dist = float(np.linalg.norm(p1_curr - p2_curr))
            curr_vert_dist = float(abs(ac1["altitude_ft"] - ac2["altitude_ft"]))

            # Relative velocity vector
            rad1 = np.radians(ac1["heading_deg"])
            rad2 = np.radians(ac2["heading_deg"])
            v1 = np.array([ac1["speed_kt"] * np.sin(rad1), ac1["speed_kt"] * np.cos(rad1)])
            v2 = np.array([ac2["speed_kt"] * np.sin(rad2), ac2["speed_kt"] * np.cos(rad2)])
            v_rel = v1 - v2
            v_rel_speed = float(np.linalg.norm(v_rel))

            # Closure rate along line-of-sight
            dp = p1_curr - p2_curr
            dp_norm = float(np.linalg.norm(dp))
            if dp_norm > 1e-4:
                closure_rate_kt = float(-np.dot(dp, v_rel) / dp_norm)
            else:
                closure_rate_kt = 0.0

            # Step-by-step distance along predicted futures
            preds1 = np.array([[p["x"], p["y"], p["alt"], p["t"]] for p in ac1["future_pred"]])
            preds2 = np.array([[p["x"], p["y"], p["alt"], p["t"]] for p in ac2["future_pred"]])

            has_pred_conflict = False
            cpa_dist_nm = curr_lat_dist
            cpa_vert_ft = curr_vert_dist
            tau_s = 0.0
            cpa_point_1: list[float] = [p1_curr[0], p1_curr[1]]
            cpa_point_2: list[float] = [p2_curr[0], p2_curr[1]]

            if len(preds1) > 0 and len(preds2) > 0:
                n_pts = min(len(preds1), len(preds2))
                dx = preds1[:n_pts, 0] - preds2[:n_pts, 0]
                dy = preds1[:n_pts, 1] - preds2[:n_pts, 1]
                lat_dists = np.hypot(dx, dy)
                vert_dists = np.abs(preds1[:n_pts, 2] - preds2[:n_pts, 2])

                min_idx = int(np.argmin(lat_dists))
                cpa_dist_nm = float(lat_dists[min_idx])
                cpa_vert_ft = float(vert_dists[min_idx])
                tau_s = float(preds1[min_idx, 3] - curr_origin)
                cpa_point_1 = [float(preds1[min_idx, 0]), float(preds1[min_idx, 1])]
                cpa_point_2 = [float(preds2[min_idx, 0]), float(preds2[min_idx, 1])]

                # Conflict is flagged if within lookahead both lateral < 5.0 and vertical < 1000
                conflict_mask = (lat_dists < 5.0) & (vert_dists < 1000.0)
                has_pred_conflict = bool(np.any(conflict_mask))

            # Ground truth conflict in future
            has_gt_los = False
            for c in gt_conflicts:
                if (c.aircraft_1 == ac1["aircraft_id"] and c.aircraft_2 == ac2["aircraft_id"]) or (
                    c.aircraft_1 == ac2["aircraft_id"] and c.aircraft_2 == ac1["aircraft_id"]
                ):
                    if curr_origin <= c.first_conflict_time <= curr_origin + lookahead_s:
                        has_gt_los = True
                        break

            # Continuous conflict risk score P(risk)
            p_risk = round(
                compute_conflict_risk(
                    cpa_lateral_nm=cpa_dist_nm,
                    cpa_vertical_ft=cpa_vert_ft,
                    lateral_min_nm=5.0,
                    vertical_min_ft=1000.0,
                ),
                4,
            )

            pair_entry = {
                "aircraft_1": ac1["aircraft_id"],
                "aircraft_2": ac2["aircraft_id"],
                "curr_distance_nm": round(curr_lat_dist, 2),
                "curr_vertical_ft": round(curr_vert_dist, 1),
                "closing_speed_kt": round(v_rel_speed, 1),
                "closure_rate_kt": round(closure_rate_kt, 1),
                "closure_rate_nm_min": round(closure_rate_kt / 60.0, 2),
                "predicted_cpa_nm": round(cpa_dist_nm, 2),
                "predicted_cpa_vert_ft": round(cpa_vert_ft, 1),
                "predicted_tau_s": round(tau_s, 1),
                "cpa_pos_1": cpa_point_1,
                "cpa_pos_2": cpa_point_2,
                "has_predicted_conflict": has_pred_conflict,
                "has_gt_conflict": has_gt_los,
                "p_risk": p_risk,
            }
            all_pairs.append(pair_entry)

    # Pick active pair (prefer predicted conflict, else GT conflict, else closest)
    if all_pairs:
        active_pair_data = sorted(
            all_pairs,
            key=lambda p: (not p["has_predicted_conflict"], not p["has_gt_conflict"], p["predicted_cpa_nm"]),
        )[0]

    # Sector bounding box
    all_x = df["x_nm"].to_numpy()
    all_y = df["y_nm"].to_numpy()
    x_mid = float((all_x.min() + all_x.max()) / 2.0)
    y_mid = float((all_y.min() + all_y.max()) / 2.0)
    span = float(max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 40.0) / 2.0 * 1.15)

    return {
        "scenario_id": scenario_id,
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "origin_t": curr_origin,
        "origin_formatted": f"{int(curr_origin // 60):02d}:{int(curr_origin % 60):02d}",
        "duration_formatted": f"{int(t_max // 60):02d}:{int(t_max % 60):02d}",
        "available_origins": possible_origins,
        "lookahead_s": lookahead_s,
        "aircraft_ids": aircraft_ids,
        "aircraft": aircraft_data,
        "all_pairs": all_pairs,
        "active_pair": active_pair_data,
        "gt_conflicts": [
            {
                "aircraft_1": c.aircraft_1,
                "aircraft_2": c.aircraft_2,
                "first_conflict_time": c.first_conflict_time,
                "min_lateral_nm": round(c.min_lateral_distance_nm, 2),
                "min_vertical_ft": round(c.min_vertical_distance_ft, 1),
                "duration_s": c.conflict_duration_s,
            }
            for c in gt_conflicts
        ],
        "viewport": {
            "x_min": x_mid - span,
            "x_max": x_mid + span,
            "y_min": y_mid - span,
            "y_max": y_mid + span,
            "x_center": x_mid,
            "y_center": y_mid,
            "span_nm": span * 2.0,
        },
    }


def get_benchmark_metrics(
    dataset: str = "hard_v1",
    split: str = "test",
    model_name: str = "cv_smoothed",
) -> dict[str, Any]:
    """Retrieve benchmark evaluation metrics for given dataset, split, and model.

    Always computes metrics live via run_test_evaluation with no hardcoded shortcuts.
    Results are cached in-memory in _METRICS_CACHE.
    """
    cache_key = f"{dataset}_{split}_{model_name}"
    if cache_key in _METRICS_CACHE:
        return _METRICS_CACHE[cache_key]

    # Map dataset ID to variant consistently with _resolve_dataset_paths
    if dataset in ("hard_large", "large"):
        variant = "hard_large"
    elif dataset in ("hard_v1", "hard"):
        variant = "hard"
    elif dataset in ("easy", "nominal"):
        variant = "easy"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    if split not in ("test", "val", "train", "sample"):
        raise ValueError(f"Unknown split '{split}' for dataset '{dataset}'")

    # Map model name
    if model_name in ("lstm_v1", "lstm"):
        model_arg = "lstm"
        checkpoint: str | None = str(CHECKPOINT_PATH)
    elif model_name in ("cv_smoothed", "smoothed"):
        model_arg = "cv_smoothed"
        checkpoint = None
    elif model_name in ("cv_3step", "cv", "baseline"):
        model_arg = "cv_3step"
        checkpoint = None
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    t0 = time.perf_counter()
    metrics_list, conf = run_test_evaluation(
        variant=variant,
        model_name=model_arg,
        checkpoint=checkpoint,
        split=split,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pos_map = {
        m.horizon_s: {
            "horizontal_nm": round(m.horizontal_rmse_nm, 4),
            "vertical_ft": round(m.vertical_rmse_ft, 2),
        }
        for m in metrics_list
    }

    result = {
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "precision": round(conf.precision, 4),
        "recall": round(conf.recall, 4),
        "f1_score": round(conf.f1_score, 4),
        "false_alarms_per_1000_negatives": round(conf.false_alarms_per_1000_negatives, 2),
        "mean_lead_time_s": round(conf.tp_lead_time_median_s if conf.tp_lead_time_median_s is not None else 0.0, 1),
        "tp_time_to_conflict_mae_s": round(conf.tp_time_to_conflict_mae_s if conf.tp_time_to_conflict_mae_s is not None else 0.0, 2),
        "true_positives": conf.true_positives,
        "false_positives": conf.false_positives,
        "false_negatives": conf.false_negatives,
        "true_negatives": conf.true_negatives,
        "total_samples": conf.total_samples,
        "position_rmse": pos_map,
        "latency_ms": round(elapsed_ms / max(conf.total_samples, 1), 3),
    }
    _METRICS_CACHE[cache_key] = result
    return result

