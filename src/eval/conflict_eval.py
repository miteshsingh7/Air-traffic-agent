"""Conflict prediction and advisory evaluation module.

Research simulation only - not for operational use.
Evaluates trajectory model conflict advisory performance across (pair, origin) windows:
- TP, FP, FN, TN
- Precision, Recall, F1
- False alarms per 1000 negative samples
- Lead-time distribution for TPs (true_conflict_time - origin_time)
- Mean absolute error between predicted and true time-to-conflict for TPs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, Sequence
import numpy as np
import pandas as pd

from src.conflict.geometry import compute_scenario_conflicts
from src.data.synthetic import TrajectoryConfig
from src.data.windows import extract_scenario_origin_groups


class TrajectoryPredictor(Protocol):
    """Protocol for trajectory prediction models."""

    def predict(self, history: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class BootstrapCIs:
    """95% Confidence intervals from scenario-level bootstrap."""

    recall: tuple[float, float]
    precision: tuple[float, float]
    f1: tuple[float, float]
    conflict_detection_rate: tuple[float, float]
    lead_conditioned_rates: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioStats:
    """Per-scenario performance statistics for scenario-level bootstrap."""

    scenario_id: str
    tp: int
    fp: int
    fn: int
    tn: int
    gt_conflict_pairs: int
    detected_conflict_pairs: int
    out_of_horizon_only_pairs: int = 0
    detected_at_lead_ge: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ConflictEvaluationResult:
    """Comprehensive conflict advisory evaluation metrics."""

    num_scenarios: int
    num_gt_conflict_pairs: int
    num_detected_conflict_pairs: int
    conflict_detection_rate: float
    first_detection_lead_min_s: float | None
    first_detection_lead_median_s: float | None
    first_detection_lead_max_s: float | None

    total_samples: int
    positive_samples: int
    negative_samples: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_alarms_per_1000_negatives: float

    tp_lead_time_min_s: float | None
    tp_lead_time_median_s: float | None
    tp_lead_time_max_s: float | None
    tp_time_to_conflict_mae_s: float | None

    out_of_horizon_only_detected_pairs: int = 0
    lead_conditioned_rates: dict[int, dict[str, Any]] = field(default_factory=dict)
    ttc_buckets: dict[str, dict[str, Any]] | None = None
    fp_categories: dict[str, int] = field(default_factory=dict)
    fp_causes: dict[str, int] = field(default_factory=dict)
    subset_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    bootstrap_cis: BootstrapCIs | None = None
    scenario_stats: list[ScenarioStats] = field(default_factory=list)

    @property
    def subgroup_metrics(self) -> dict[str, dict[str, Any]]:
        """Alias for backward compatibility with earlier code."""
        return self.subset_metrics


def compute_scenario_bootstrap_cis(
    scenario_stats: Sequence[ScenarioStats],
    n_bootstraps: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> BootstrapCIs:
    """Compute 95% bootstrap confidence intervals resampled at the SCENARIO level.

    Whole scenarios are resampled with replacement to strictly preserve
    intra-scenario spatial and temporal correlations across aircraft pairs and origins.
    """
    n_scenarios = len(scenario_stats)
    if n_scenarios == 0:
        return BootstrapCIs(
            recall=(0.0, 0.0),
            precision=(0.0, 0.0),
            f1=(0.0, 0.0),
            conflict_detection_rate=(0.0, 0.0),
            lead_conditioned_rates={L: (0.0, 0.0) for L in (60, 120, 180, 240)},
        )

    arr_tp = np.array([s.tp for s in scenario_stats], dtype=np.int64)
    arr_fp = np.array([s.fp for s in scenario_stats], dtype=np.int64)
    arr_fn = np.array([s.fn for s in scenario_stats], dtype=np.int64)
    arr_gt = np.array([s.gt_conflict_pairs for s in scenario_stats], dtype=np.int64)
    arr_det = np.array([s.detected_conflict_pairs for s in scenario_stats], dtype=np.int64)

    rng = np.random.default_rng(seed)
    # Shape: (n_bootstraps, n_scenarios) - resamples scenario indices
    idx = rng.integers(0, n_scenarios, size=(n_bootstraps, n_scenarios))

    boot_tp = np.sum(arr_tp[idx], axis=1)
    boot_fp = np.sum(arr_fp[idx], axis=1)
    boot_fn = np.sum(arr_fn[idx], axis=1)
    boot_gt = np.sum(arr_gt[idx], axis=1)
    boot_det = np.sum(arr_det[idx], axis=1)

    denom_p = boot_tp + boot_fp
    safe_denom_p = np.where(denom_p > 0, denom_p, 1)
    boot_precision = np.where(denom_p > 0, boot_tp / safe_denom_p, 0.0)

    denom_r = boot_tp + boot_fn
    safe_denom_r = np.where(denom_r > 0, denom_r, 1)
    boot_recall = np.where(denom_r > 0, boot_tp / safe_denom_r, 0.0)

    denom_f1 = boot_precision + boot_recall
    safe_denom_f1 = np.where(denom_f1 > 0, denom_f1, 1.0)
    boot_f1 = np.where(denom_f1 > 0, 2.0 * boot_precision * boot_recall / safe_denom_f1, 0.0)

    denom_det = boot_gt
    safe_denom_det = np.where(denom_det > 0, denom_det, 1)
    boot_det_rate = np.where(denom_det > 0, boot_det / safe_denom_det, 0.0)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = 100.0 * alpha
    upper_pct = 100.0 * (1.0 - alpha)

    lead_cis: dict[int, tuple[float, float]] = {}
    for L in (60, 120, 180, 240):
        arr_lead = np.array(
            [s.detected_at_lead_ge.get(L, 0) for s in scenario_stats], dtype=np.int64
        )
        boot_lead = np.sum(arr_lead[idx], axis=1)
        boot_lead_rate = np.where(denom_det > 0, boot_lead / safe_denom_det, 0.0)
        lead_cis[L] = (
            float(np.percentile(boot_lead_rate, lower_pct)),
            float(np.percentile(boot_lead_rate, upper_pct)),
        )

    return BootstrapCIs(
        recall=(
            float(np.percentile(boot_recall, lower_pct)),
            float(np.percentile(boot_recall, upper_pct)),
        ),
        precision=(
            float(np.percentile(boot_precision, lower_pct)),
            float(np.percentile(boot_precision, upper_pct)),
        ),
        f1=(
            float(np.percentile(boot_f1, lower_pct)),
            float(np.percentile(boot_f1, upper_pct)),
        ),
        conflict_detection_rate=(
            float(np.percentile(boot_det_rate, lower_pct)),
            float(np.percentile(boot_det_rate, upper_pct)),
        ),
        lead_conditioned_rates=lead_cis,
    )


def evaluate_conflict_prediction(
    scenario_files: Sequence[str | Path],
    predictor: TrajectoryPredictor,
    lateral_min_nm: float = 5.0,
    vertical_min_ft: float = 1000.0,
    origin_step_s: float = 30.0,
    history_s: float = 60.0,
    horizon_s: float = 300.0,
    dt_s: float = 5.0,
    use_noisy_history: bool = True,
    metadata: dict[str, Any] | None = None,
    n_bootstraps: int = 1000,
    bootstrap_seed: int = 42,
) -> ConflictEvaluationResult:
    """Evaluate conflict advisory predictions against ground-truth future encounters.

    Computes:
    - Number of scenarios and distinct ground-truth conflict pairs
    - Conflict-level metrics: detection rate & lead time of first detection
    - Window-level classification metrics: TP, FP, FN, TN, precision, recall, F1
    - False-positive breakdown by scenario category and primary cause
    - Subset metrics over all samples: maneuvering, non-maneuvering, near-miss, other-clean
    - 95% bootstrap confidence intervals resampled at scenario level.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    tp_lead_times: list[float] = []
    tp_time_errors: list[float] = []

    bucket_keys = ["0-60", "60-120", "120-180", "180-240", "240-300"]
    bucket_pos = {k: 0 for k in bucket_keys}
    bucket_tp = {k: 0 for k in bucket_keys}

    fp_categories = {
        "injected_conflict": 0,
        "near_miss": 0,
        "other_clean": 0,
    }
    fp_causes = {
        "lateral": 0,
        "vertical": 0,
        "both": 0,
    }

    subset_counts = {
        "maneuvering": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "non_maneuvering": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "near_miss": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "other_clean": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    # Conflict-level tracking across all scenarios
    all_first_detection_leads: list[float] = []
    scenario_stats_list: list[ScenarioStats] = []

    future_steps = int(round(horizon_s / dt_s))

    # Auto-load metadata if omitted and metadata.json exists in directory of scenario_files[0]
    loaded_metadata = metadata
    if loaded_metadata is None and len(scenario_files) > 0:
        candidate_meta = Path(scenario_files[0]).parent / "metadata.json"
        if candidate_meta.exists():
            try:
                with open(candidate_meta, "r", encoding="utf-8") as f:
                    loaded_metadata = json.load(f)
            except Exception:
                loaded_metadata = None

    for file_path in scenario_files:
        p = Path(file_path)
        scenario_id = p.stem
        scen_meta = loaded_metadata.get(scenario_id, {}) if loaded_metadata else {}

        is_injected_scen = bool(scen_meta.get("injected_conflict", False))
        is_maneuvering_scen = bool(scen_meta.get("maneuvering", False))
        is_non_maneuvering_scen = is_injected_scen and not is_maneuvering_scen
        is_near_miss_scen = (not is_injected_scen) and bool(scen_meta.get("near_miss", False))
        is_other_clean_scen = (not is_injected_scen) and (not is_near_miss_scen)

        if is_maneuvering_scen:
            subset_key = "maneuvering"
        elif is_non_maneuvering_scen:
            subset_key = "non_maneuvering"
        elif is_near_miss_scen:
            subset_key = "near_miss"
        else:
            subset_key = "other_clean"

        df = pd.read_parquet(file_path)

        # 1. Distinct ground-truth conflict pairs and earliest onset times in this scenario
        scenario_conflicts = compute_scenario_conflicts(
            df,
            lateral_min_nm=lateral_min_nm,
            vertical_min_ft=vertical_min_ft,
        )
        gt_pairs: dict[tuple[str, str], float] = {}
        for c in scenario_conflicts:
            pair_key = (min(c.aircraft_1, c.aircraft_2), max(c.aircraft_1, c.aircraft_2))
            if pair_key not in gt_pairs or c.first_conflict_time < gt_pairs[pair_key]:
                gt_pairs[pair_key] = c.first_conflict_time

        detected_in_horizon_pairs: set[tuple[str, str]] = set()
        detected_out_of_horizon_pairs: set[tuple[str, str]] = set()
        first_detection_origins: dict[tuple[str, str], float] = {}
        lead_detected_pairs: dict[int, set[tuple[str, str]]] = {
            60: set(),
            120: set(),
            180: set(),
            240: set(),
        }

        scen_tp = 0
        scen_fp = 0
        scen_fn = 0
        scen_tn = 0

        origin_groups = extract_scenario_origin_groups(
            scenario_df=df,
            origin_step_s=origin_step_s,
            history_s=history_s,
            horizon_s=horizon_s,
            dt_s=dt_s,
            use_noisy_history=use_noisy_history,
        )

        for group in origin_groups:
            origin_t = group.origin_t
            aircraft_ids = group.aircraft_ids
            num_ac = len(aircraft_ids)
            if num_ac < 2:
                continue

            # Model predictions for all aircraft at this origin
            pred_futures: dict[str, np.ndarray] = {
                acid: predictor.predict(group.history[acid]) for acid in aircraft_ids
            }

            fut_times = np.array(
                [origin_t + (k + 1) * dt_s for k in range(future_steps)],
                dtype=np.float64,
            )

            for i in range(num_ac):
                acid_1 = aircraft_ids[i]
                true_1 = group.future_true[acid_1]
                pred_1 = pred_futures[acid_1]

                for j in range(i + 1, num_ac):
                    acid_2 = aircraft_ids[j]
                    true_2 = group.future_true[acid_2]
                    pred_2 = pred_futures[acid_2]
                    pair_key = (min(acid_1, acid_2), max(acid_1, acid_2))

                    # Ground truth future geometry
                    dx_true = true_1[:, 0] - true_2[:, 0]
                    dy_true = true_1[:, 1] - true_2[:, 1]
                    dz_true = true_1[:, 2] - true_2[:, 2]
                    lat_true = np.hypot(dx_true, dy_true)
                    vert_true = np.abs(dz_true)

                    true_conflict_mask = (lat_true < lateral_min_nm) & (vert_true < vertical_min_ft)
                    has_true_conflict = bool(np.any(true_conflict_mask))

                    true_first_t: float | None = None
                    true_ttc: float | None = None
                    if has_true_conflict:
                        first_idx_true = int(np.where(true_conflict_mask)[0][0])
                        true_first_t = float(fut_times[first_idx_true])
                        true_ttc = true_first_t - origin_t

                    # Predicted future geometry
                    dx_pred = pred_1[:, 0] - pred_2[:, 0]
                    dy_pred = pred_1[:, 1] - pred_2[:, 1]
                    dz_pred = pred_1[:, 2] - pred_2[:, 2]
                    lat_pred = np.hypot(dx_pred, dy_pred)
                    vert_pred = np.abs(dz_pred)

                    pred_conflict_mask = (lat_pred < lateral_min_nm) & (vert_pred < vertical_min_ft)
                    has_pred_conflict = bool(np.any(pred_conflict_mask))

                    pred_first_t: float | None = None
                    pred_ttc: float | None = None
                    if has_pred_conflict:
                        first_idx_pred = int(np.where(pred_conflict_mask)[0][0])
                        pred_first_t = float(fut_times[first_idx_pred])
                        pred_ttc = pred_first_t - origin_t

                    # Conflict-level tracking:
                    if pair_key in gt_pairs:
                        true_onset = gt_pairs[pair_key]
                        lead_s = true_onset - origin_t
                        if origin_t < true_onset and has_pred_conflict:
                            if lead_s <= horizon_s:
                                detected_in_horizon_pairs.add(pair_key)
                                if pair_key not in first_detection_origins:
                                    first_detection_origins[pair_key] = origin_t
                                else:
                                    first_detection_origins[pair_key] = min(
                                        first_detection_origins[pair_key], origin_t
                                    )
                                for L in (60, 120, 180, 240):
                                    if L <= lead_s <= horizon_s:
                                        lead_detected_pairs[L].add(pair_key)
                            else:
                                detected_out_of_horizon_pairs.add(pair_key)

                    # Window-level sample classification
                    if has_true_conflict and has_pred_conflict:
                        tp += 1
                        scen_tp += 1
                        subset_counts[subset_key]["tp"] += 1
                        assert true_ttc is not None and pred_ttc is not None
                        tp_lead_times.append(true_ttc)
                        tp_time_errors.append(abs(pred_ttc - true_ttc))
                    elif not has_true_conflict and has_pred_conflict:
                        fp += 1
                        scen_fp += 1
                        subset_counts[subset_key]["fp"] += 1

                        # FP Breakdown 1: Category
                        if is_injected_scen:
                            fp_categories["injected_conflict"] += 1
                        elif is_near_miss_scen:
                            fp_categories["near_miss"] += 1
                        else:
                            fp_categories["other_clean"] += 1

                        # FP Breakdown 2: Primary Cause
                        min_lat_true = float(np.min(lat_true))
                        min_vert_true = float(np.min(vert_true))
                        if min_lat_true >= lateral_min_nm and min_vert_true < vertical_min_ft:
                            fp_causes["lateral"] += 1
                        elif min_lat_true < lateral_min_nm and min_vert_true >= vertical_min_ft:
                            fp_causes["vertical"] += 1
                        elif min_lat_true >= lateral_min_nm and min_vert_true >= vertical_min_ft:
                            fp_causes["both"] += 1
                        else:
                            k_los = int(np.where(pred_conflict_mask)[0][0])
                            lat_sep = lat_true[k_los] >= lateral_min_nm
                            vert_sep = vert_true[k_los] >= vertical_min_ft
                            if lat_sep and not vert_sep:
                                fp_causes["lateral"] += 1
                            elif vert_sep and not lat_sep:
                                fp_causes["vertical"] += 1
                            elif lat_sep and vert_sep:
                                fp_causes["both"] += 1
                            else:
                                err_lat = abs(lat_pred[k_los] - lat_true[k_los]) / lateral_min_nm
                                err_vert = abs(vert_pred[k_los] - vert_true[k_los]) / vertical_min_ft
                                if err_lat >= err_vert:
                                    fp_causes["lateral"] += 1
                                else:
                                    fp_causes["vertical"] += 1

                    elif has_true_conflict and not has_pred_conflict:
                        fn += 1
                        scen_fn += 1
                        subset_counts[subset_key]["fn"] += 1
                    else:
                        tn += 1
                        scen_tn += 1
                        subset_counts[subset_key]["tn"] += 1

                    # TTC bucket tracking
                    if has_true_conflict and true_ttc is not None:
                        if true_ttc <= 60.0:
                            b_key = "0-60"
                        elif true_ttc <= 120.0:
                            b_key = "60-120"
                        elif true_ttc <= 180.0:
                            b_key = "120-180"
                        elif true_ttc <= 240.0:
                            b_key = "180-240"
                        else:
                            b_key = "240-300"
                        bucket_pos[b_key] += 1
                        if has_pred_conflict:
                            bucket_tp[b_key] += 1

        # Out-of-horizon only pairs: detected with lead > horizon_s, but never detected with lead <= horizon_s
        out_of_horizon_only_set = detected_out_of_horizon_pairs - detected_in_horizon_pairs
        scen_out_of_horizon_only = len(out_of_horizon_only_set)

        # Record lead times of first detection for in-horizon detected GT pairs in this scenario
        for pair_key in detected_in_horizon_pairs:
            true_onset = gt_pairs[pair_key]
            first_orig = first_detection_origins[pair_key]
            lead_s = true_onset - first_orig
            all_first_detection_leads.append(lead_s)

        scen_lead_ge_counts = {L: len(lead_detected_pairs[L]) for L in (60, 120, 180, 240)}

        scenario_stats_list.append(
            ScenarioStats(
                scenario_id=scenario_id,
                tp=scen_tp,
                fp=scen_fp,
                fn=scen_fn,
                tn=scen_tn,
                gt_conflict_pairs=len(gt_pairs),
                detected_conflict_pairs=len(detected_in_horizon_pairs),
                out_of_horizon_only_pairs=scen_out_of_horizon_only,
                detected_at_lead_ge=scen_lead_ge_counts,
            )
        )

    # Aggregate conflict-level metrics
    num_scenarios = len(scenario_files)
    num_gt_conflict_pairs = sum(s.gt_conflict_pairs for s in scenario_stats_list)
    num_detected_conflict_pairs = sum(s.detected_conflict_pairs for s in scenario_stats_list)
    num_out_of_horizon_only = sum(s.out_of_horizon_only_pairs for s in scenario_stats_list)
    conflict_detection_rate = (
        float(num_detected_conflict_pairs / num_gt_conflict_pairs)
        if num_gt_conflict_pairs > 0
        else 0.0
    )

    det_lead_min = float(np.min(all_first_detection_leads)) if all_first_detection_leads else None
    det_lead_median = float(np.median(all_first_detection_leads)) if all_first_detection_leads else None
    det_lead_max = float(np.max(all_first_detection_leads)) if all_first_detection_leads else None

    # Aggregate sample-level metrics
    total_samples = tp + fp + fn + tn
    positive_samples = tp + fn
    negative_samples = fp + tn

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    fa_per_1000 = float(fp / negative_samples * 1000.0) if negative_samples > 0 else 0.0

    tp_lead_min = float(np.min(tp_lead_times)) if tp_lead_times else None
    tp_lead_median = float(np.median(tp_lead_times)) if tp_lead_times else None
    tp_lead_max = float(np.max(tp_lead_times)) if tp_lead_times else None
    mae_ttc = float(np.mean(tp_time_errors)) if tp_time_errors else None

    # TTC Buckets
    ttc_buckets: dict[str, dict[str, Any]] = {}
    for k in bucket_keys:
        p_cnt = bucket_pos[k]
        t_cnt = bucket_tp[k]
        r_val = float(t_cnt / p_cnt) if p_cnt > 0 else 0.0
        ttc_buckets[k] = {"pos": p_cnt, "tp": t_cnt, "recall": r_val}

    # Subset metrics over all samples
    subset_metrics: dict[str, dict[str, Any]] = {}
    for sub in ("maneuvering", "non_maneuvering", "near_miss", "other_clean"):
        s_tp = subset_counts[sub]["tp"]
        s_fp = subset_counts[sub]["fp"]
        s_fn = subset_counts[sub]["fn"]
        s_tn = subset_counts[sub]["tn"]
        s_total = s_tp + s_fp + s_fn + s_tn
        s_pos = s_tp + s_fn
        s_neg = s_fp + s_tn
        s_prec = float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) > 0 else 0.0
        s_rec = float(s_tp / (s_tp + s_fn)) if (s_tp + s_fn) > 0 else 0.0
        s_fa_rate = float(s_fp / s_neg * 1000.0) if s_neg > 0 else 0.0
        subset_metrics[sub] = {
            "total_samples": s_total,
            "pos": s_pos,
            "neg": s_neg,
            "tp": s_tp,
            "fp": s_fp,
            "fn": s_fn,
            "tn": s_tn,
            "precision": s_prec,
            "recall": s_rec,
            "fa_per_1000": s_fa_rate,
        }

    # Scenario-level Bootstrap Confidence Intervals
    bootstrap_cis = compute_scenario_bootstrap_cis(
        scenario_stats=scenario_stats_list,
        n_bootstraps=n_bootstraps,
        seed=bootstrap_seed,
    )

    lead_conditioned_rates: dict[int, dict[str, Any]] = {}
    for L in (60, 120, 180, 240):
        c_l = sum(s.detected_at_lead_ge.get(L, 0) for s in scenario_stats_list)
        r_l = float(c_l / num_gt_conflict_pairs) if num_gt_conflict_pairs > 0 else 0.0
        ci_l = bootstrap_cis.lead_conditioned_rates.get(L, (0.0, 0.0))
        lead_conditioned_rates[L] = {"count": c_l, "rate": r_l, "ci_95": ci_l}

    return ConflictEvaluationResult(
        num_scenarios=num_scenarios,
        num_gt_conflict_pairs=num_gt_conflict_pairs,
        num_detected_conflict_pairs=num_detected_conflict_pairs,
        conflict_detection_rate=conflict_detection_rate,
        first_detection_lead_min_s=det_lead_min,
        first_detection_lead_median_s=det_lead_median,
        first_detection_lead_max_s=det_lead_max,
        total_samples=total_samples,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        false_alarms_per_1000_negatives=fa_per_1000,
        tp_lead_time_min_s=tp_lead_min,
        tp_lead_time_median_s=tp_lead_median,
        tp_lead_time_max_s=tp_lead_max,
        tp_time_to_conflict_mae_s=mae_ttc,
        out_of_horizon_only_detected_pairs=num_out_of_horizon_only,
        lead_conditioned_rates=lead_conditioned_rates,
        ttc_buckets=ttc_buckets,
        fp_categories=fp_categories,
        fp_causes=fp_causes,
        subset_metrics=subset_metrics,
        bootstrap_cis=bootstrap_cis,
        scenario_stats=scenario_stats_list,
    )


def format_conflict_evaluation(res: ConflictEvaluationResult) -> str:
    """Format ConflictEvaluationResult into a readable summary string."""
    ci = res.bootstrap_cis

    ci_rec_str = f"[{ci.recall[0]:.4f}, {ci.recall[1]:.4f}]" if ci else "N/A"
    ci_prec_str = f"[{ci.precision[0]:.4f}, {ci.precision[1]:.4f}]" if ci else "N/A"
    ci_f1_str = f"[{ci.f1[0]:.4f}, {ci.f1[1]:.4f}]" if ci else "N/A"
    ci_det_str = (
        f"[{ci.conflict_detection_rate[0]:.4f}, {ci.conflict_detection_rate[1]:.4f}]"
        if ci
        else "N/A"
    )

    first_lead_str = (
        f"min={res.first_detection_lead_min_s:.1f} s, "
        f"median={res.first_detection_lead_median_s:.1f} s, "
        f"max={res.first_detection_lead_max_s:.1f} s"
        if res.first_detection_lead_min_s is not None
        else "N/A (0 detected pairs)"
    )

    tp_lead_str = (
        f"min={res.tp_lead_time_min_s:.1f} s, "
        f"median={res.tp_lead_time_median_s:.1f} s, "
        f"max={res.tp_lead_time_max_s:.1f} s"
        if res.tp_lead_time_min_s is not None
        else "N/A (no TPs)"
    )
    mae_str = (
        f"{res.tp_time_to_conflict_mae_s:.2f} s"
        if res.tp_time_to_conflict_mae_s is not None
        else "N/A (no TPs)"
    )

    lines: list[str] = [
        "----------------------------------------------------------------------",
        "CONFLICT PREDICTION & ADVISORY EVALUATION",
        "----------------------------------------------------------------------",
        f"Test Scenarios Evaluated        : {res.num_scenarios}",
        f"Distinct Ground-Truth Conflicts : {res.num_gt_conflict_pairs} pairs",
        "",
        "Conflict-Level Detection Performance:",
        f"  - Detected Conflict Pairs (lead <= 300 s): {res.num_detected_conflict_pairs} / {res.num_gt_conflict_pairs}",
        f"  - Conflict Detection Rate     : {res.conflict_detection_rate:.4f}  (95% CI: {ci_det_str})",
        f"  - Out-of-Horizon Only Detected: {res.out_of_horizon_only_detected_pairs} pairs (detected only at lead > 300 s)",
        f"  - First-Detection Lead Time   : {first_lead_str}",
        "",
        "Lead-Conditioned Conflict Detection Rate (lead >= L within 300 s horizon):",
    ]

    for L in (60, 120, 180, 240):
        lr = res.lead_conditioned_rates.get(L, {})
        c_l = lr.get("count", 0)
        r_l = lr.get("rate", 0.0)
        ci_l = lr.get("ci_95", (0.0, 0.0))
        lines.append(
            f"  - Lead >= {L:>3} s                 : {c_l:>3} / {res.num_gt_conflict_pairs} = {r_l:.4f}  "
            f"(95% CI: [{ci_l[0]:.4f}, {ci_l[1]:.4f}])"
        )

    lines.extend([
        "",
        "Sample-Level Classification (pair, origin):",
        f"  - Total (pair, origin) samples: {res.total_samples}",
        f"  - Positive samples (LoS true) : {res.positive_samples}",
        f"  - Negative samples (LoS false): {res.negative_samples}",
        "",
        "Confusion Matrix:",
        f"  - True Positives  (TP)        : {res.true_positives}",
        f"  - False Positives (FP)        : {res.false_positives}",
        f"  - False Negatives (FN)        : {res.false_negatives}",
        f"  - True Negatives  (TN)        : {res.true_negatives}",
        "",
        "Sample-Level Performance & 95% Bootstrap CIs (Scenario-Level Resampling):",
        f"  - Precision                   : {res.precision:.4f}  (95% CI: {ci_prec_str})",
        f"  - Recall                      : {res.recall:.4f}  (95% CI: {ci_rec_str})",
        f"  - F1 Score                    : {res.f1_score:.4f}  (95% CI: {ci_f1_str})",
        f"  - False Alarms per 1000 Neg   : {res.false_alarms_per_1000_negatives:.2f}",
        "",
        "False Positive Breakdown:",
        f"  By Scenario Category (sum={sum(res.fp_categories.values())} / total_fp={res.false_positives}):",
        f"    - Injected-conflict scenarios : {res.fp_categories.get('injected_conflict', 0)}",
        f"    - Near-miss scenarios         : {res.fp_categories.get('near_miss', 0)}",
        f"    - Other clean scenarios       : {res.fp_categories.get('other_clean', 0)}",
        f"  By Primary Cause (sum={sum(res.fp_causes.values())} / total_fp={res.false_positives}):",
        f"    - Lateral error               : {res.fp_causes.get('lateral', 0)}",
        f"    - Vertical error              : {res.fp_causes.get('vertical', 0)}",
        f"    - Both lateral & vertical     : {res.fp_causes.get('both', 0)}",
        "",
        "Performance by Scenario Subset (All Samples):",
    ])

    for sub, name in [
        ("maneuvering", "Maneuvering Conflict Scenarios"),
        ("non_maneuvering", "Non-Maneuvering Conflict Scenarios"),
        ("near_miss", "Near-Miss Scenarios"),
        ("other_clean", "Other Clean Scenarios"),
    ]:
        sm = res.subset_metrics.get(sub, {})
        tot = sm.get("total_samples", 0)
        pos = sm.get("pos", 0)
        neg = sm.get("neg", 0)
        s_tp = sm.get("tp", 0)
        s_fp = sm.get("fp", 0)
        s_fn = sm.get("fn", 0)
        s_tn = sm.get("tn", 0)
        fa_rate = sm.get("fa_per_1000", 0.0)

        if pos == 0:
            lines.append(
                f"  - {name:<35}: N={tot:<5} (pos={pos:<4}, neg={neg:<5}) | "
                f"recall=N/A, FA/1000 neg={fa_rate:.2f} | TP={s_tp}, FP={s_fp}, FN={s_fn}, TN={s_tn}"
            )
        else:
            rec = sm.get("recall", 0.0)
            prec = sm.get("precision", 0.0)
            lines.append(
                f"  - {name:<35}: N={tot:<5} (pos={pos:<4}, neg={neg:<5}) | "
                f"recall={rec:.4f}, precision={prec:.4f}, FA/1000 neg={fa_rate:.2f} | TP={s_tp}, FP={s_fp}, FN={s_fn}, TN={s_tn}"
            )

    lines.extend([
        "",
        "True Positive Lead-Time & TTC Accuracy (Window-Level):",
        f"  - Window Lead-time dist       : {tp_lead_str}",
        f"  - Mean Absolute TTC Error     : {mae_str}",
    ])

    if res.ttc_buckets:
        lines.append("")
        lines.append("Recall by True Time-to-Conflict (TTC) Bucket:")
        for k in ("0-60", "60-120", "120-180", "180-240", "240-300"):
            b = res.ttc_buckets.get(k, {})
            lines.append(f"  - {k:>7} s : pos={b.get('pos', 0):<4} tp={b.get('tp', 0):<4} recall={b.get('recall', 0.0):.4f}")

    lines.append("----------------------------------------------------------------------")
    return "\n".join(lines)


def main() -> None:
    """CLI runner for conflict evaluation on any dataset via --data-dir."""
    from src.data.splits import get_or_create_splits
    from src.models.baseline import ConstantVelocityPredictor, SmoothedConstantVelocityPredictor

    parser = argparse.ArgumentParser(description="Evaluate conflict advisory predictions.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing dataset Parquet files")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--variant", type=str, default=None, help="Dataset variant (default: auto-detected)")
    parser.add_argument(
        "--model",
        type=str,
        default="cv_3step",
        choices=["cv_3step", "cv_smoothed", "lstm"],
        help="Predictor baseline model",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file for --model lstm")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (default: test)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    variant = args.variant
    if variant is None:
        if "hard_large" in data_dir.name:
            variant = "hard_large"
        elif "hard" in data_dir.name:
            variant = "hard"
        else:
            variant = "easy"

    cfg = TrajectoryConfig.from_yaml(args.config, variant=variant)

    splits = get_or_create_splits(
        data_dir=data_dir,
        config_path=args.config,
        variant=variant,
        output_file=cfg.splits_file,
    )
    target_ids = splits.get(args.split, [])
    scenario_files = [data_dir / f"{sid}.parquet" for sid in target_ids if (data_dir / f"{sid}.parquet").exists()]
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found for split {args.split} in {data_dir}")

    meta_path = data_dir / "metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    horizon_steps = int(round(cfg.horizon_s / cfg.resample_rate_s))
    if args.model == "lstm":
        from src.models.lstm_predictor import LSTMTrajectoryPredictor
        if not args.checkpoint:
            raise ValueError("--checkpoint must be provided when --model lstm is selected")
        predictor: TrajectoryPredictor = LSTMTrajectoryPredictor.from_checkpoint(args.checkpoint)
    elif args.model == "cv_smoothed":
        predictor = SmoothedConstantVelocityPredictor(
            dt_s=cfg.resample_rate_s,
            horizon_steps=horizon_steps,
        )
    else:
        predictor = ConstantVelocityPredictor(
            dt_s=cfg.resample_rate_s,
            horizon_steps=horizon_steps,
        )

    result = evaluate_conflict_prediction(
        scenario_files=scenario_files,
        predictor=predictor,
        lateral_min_nm=cfg.lateral_min_nm,
        vertical_min_ft=cfg.vertical_min_ft,
        origin_step_s=30.0,
        history_s=cfg.history_s,
        horizon_s=cfg.horizon_s,
        dt_s=cfg.resample_rate_s,
        metadata=metadata,
    )

    print(format_conflict_evaluation(result))


if __name__ == "__main__":
    main()
