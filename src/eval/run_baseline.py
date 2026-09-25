"""Run evaluation harness and constant-velocity baseline on the TEST split.

Research simulation only - not for operational use.
Evaluates:
1. Trajectory position errors (Horizontal RMSE in NM, Vertical RMSE in ft)
   at forecast horizons 30, 60, 120, 180, 240, 300 s.
2. Conflict advisory performance on (pair, origin) samples:
   - Positive/negative samples, TP/FP/FN/TN, Precision, Recall, F1
   - False alarms per 1000 negative samples
   - Lead-time distribution for TPs (min, median, max)
   - Mean absolute error between predicted and true time-to-conflict for TPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.data.splits import get_or_create_splits
from src.data.synthetic import TrajectoryConfig
from src.data.windows import extract_dataset_windows
from src.eval.conflict_eval import ConflictEvaluationResult, evaluate_conflict_prediction, format_conflict_evaluation
from src.eval.metrics import PositionErrorMetric, evaluate_position_errors, format_position_errors
from src.models.baseline import ConstantVelocityPredictor, SmoothedConstantVelocityPredictor


def run_test_evaluation(
    data_dir: str | Path | None = None,
    config_path: str | Path = "configs/default.yaml",
    variant: str = "easy",
    model_name: str = "cv_3step",
    checkpoint: str | Path | None = None,
    split: str = "test",
) -> tuple[list[PositionErrorMetric], ConflictEvaluationResult]:
    """Run full baseline evaluation pipeline on the specified split."""
    cfg = TrajectoryConfig.from_yaml(config_path, variant=variant)
    target_data_dir = Path(data_dir or cfg.output_dir)

    # 1. Retrieve or generate dataset splits
    splits = get_or_create_splits(
        data_dir=target_data_dir,
        config_path=config_path,
        variant=variant,
        output_file=cfg.splits_file,
    )
    target_scenario_ids = splits.get(split, [])

    target_files = [
        target_data_dir / f"{sid}.parquet"
        for sid in target_scenario_ids
        if (target_data_dir / f"{sid}.parquet").exists()
    ]
    if not target_files:
        raise FileNotFoundError(f"No scenario files found in {target_data_dir} for split '{split}'")

    horizon_steps = int(round(cfg.horizon_s / cfg.resample_rate_s))
    if model_name == "lstm":
        from src.models.lstm_predictor import LSTMTrajectoryPredictor
        if not checkpoint:
            raise ValueError("checkpoint path must be provided when model_name is 'lstm'")
        predictor = LSTMTrajectoryPredictor.from_checkpoint(checkpoint)
        model_display = f"RESIDUAL LSTM ({Path(checkpoint).name})"
    elif model_name in ("cv_smoothed", "smoothed"):
        predictor = SmoothedConstantVelocityPredictor(
            dt_s=cfg.resample_rate_s,
            horizon_steps=horizon_steps,
        )
        model_display = "SMOOTHED CONSTANT-VELOCITY (cv_smoothed)"
    else:
        predictor = ConstantVelocityPredictor(
            dt_s=cfg.resample_rate_s,
            horizon_steps=horizon_steps,
        )
        model_display = "3-STEP CONSTANT-VELOCITY (cv_3step)"

    total_scenarios = len(splits['train']) + len(splits['val']) + len(splits['test'])
    pct = (len(target_files) / total_scenarios * 100.0) if total_scenarios > 0 else 0.0
    print("=" * 70)
    print(f"BASELINE EVALUATION: {model_display} ({variant.upper()} DATASET - {split.upper()} SPLIT)")
    print("=" * 70)
    print(f"Total dataset scenarios       : {total_scenarios}")
    print(f"{split.capitalize()} split scenarios evaluated: {len(target_files)} ({pct:.1f}% scenario-level split)")
    print(f"Separation minima             : {cfg.lateral_min_nm} NM lateral, {cfg.vertical_min_ft} ft vertical")
    print(f"Observation / Horizon         : history={cfg.history_s}s, horizon={cfg.horizon_s}s, dt={cfg.resample_rate_s}s")
    print("=" * 70)

    # 2. Trajectory Position Error Evaluation
    print(f"\n[PART 1: TRAJECTORY POSITION PREDICTION ERROR - {variant.upper()} - {model_name}]")
    histories, futures_true, metadata = extract_dataset_windows(
        scenario_files=target_files,
        origin_step_s=30.0,
        history_s=cfg.history_s,
        horizon_s=cfg.horizon_s,
        dt_s=cfg.resample_rate_s,
        use_noisy_history=True,
    )
    print(f"Total window samples extracted: {len(metadata)}")

    futures_pred = predictor.predict(histories)
    horizons = (30, 60, 120, 180, 240, 300)
    metrics = evaluate_position_errors(
        y_pred=futures_pred,
        y_true=futures_true,
        horizons_s=horizons,
        dt_s=cfg.resample_rate_s,
    )
    print(format_position_errors(metrics))

    # 3. Conflict Advisory Evaluation
    print(f"\n[PART 2: CONFLICT ADVISORY & SEPARATION RISK EVALUATION - {variant.upper()} - {model_name}]")
    meta_path = target_data_dir / "metadata.json"
    scen_metadata = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            scen_metadata = json.load(f)

    conflict_result = evaluate_conflict_prediction(
        scenario_files=target_files,
        predictor=predictor,
        lateral_min_nm=cfg.lateral_min_nm,
        vertical_min_ft=cfg.vertical_min_ft,
        origin_step_s=30.0,
        history_s=cfg.history_s,
        horizon_s=cfg.horizon_s,
        dt_s=cfg.resample_rate_s,
        metadata=scen_metadata,
    )
    print(format_conflict_evaluation(conflict_result))

    return metrics, conflict_result


def format_comparison_table(
    easy_metrics: list[PositionErrorMetric],
    easy_conf: ConflictEvaluationResult,
    hard_3step_metrics: list[PositionErrorMetric],
    hard_3step_conf: ConflictEvaluationResult,
    hard_smoothed_metrics: list[PositionErrorMetric],
    hard_smoothed_conf: ConflictEvaluationResult,
) -> str:
    """Format a clean side-by-side comparison table between easy and hard test evaluations."""
    easy_map = {m.horizon_s: m for m in easy_metrics}
    h3_map = {m.horizon_s: m for m in hard_3step_metrics}
    hs_map = {m.horizon_s: m for m in hard_smoothed_metrics}

    col1_w = 34
    col2_w = 18
    col3_w = 18
    col4_w = 20

    header = (
        f"{'Metric':<{col1_w}} | {'cv_3step (Easy)':<{col2_w}} | "
        f"{'cv_3step (Hard)':<{col3_w}} | {'cv_smoothed (Hard)':<{col4_w}}"
    )
    sep_line = "-" * col1_w + "-+-" + "-" * col2_w + "-+-" + "-" * col3_w + "-+-" + "-" * col4_w

    lines = [
        "=" * len(header),
        "SIDE-BY-SIDE BENCHMARK COMPARISON: cv_3step vs cv_smoothed (TEST SPLIT)",
        "=" * len(header),
        header,
        sep_line,
    ]

    # Horizontal RMSE per horizon
    for hz in (30, 60, 120, 180, 240, 300):
        em = easy_map.get(hz)
        h3m = h3_map.get(hz)
        hsm = hs_map.get(hz)
        e_val = f"{em.horizontal_rmse_nm:.4f} NM" if em else "N/A"
        h3_val = f"{h3m.horizontal_rmse_nm:.4f} NM" if h3m else "N/A"
        hs_val = f"{hsm.horizontal_rmse_nm:.4f} NM" if hsm else "N/A"
        lines.append(f"{f'Horizontal RMSE @ {hz}s':<{col1_w}} | {e_val:<{col2_w}} | {h3_val:<{col3_w}} | {hs_val:<{col4_w}}")

    lines.append(sep_line)

    # Vertical RMSE per horizon
    for hz in (30, 60, 120, 180, 240, 300):
        em = easy_map.get(hz)
        h3m = h3_map.get(hz)
        hsm = hs_map.get(hz)
        e_val = f"{em.vertical_rmse_ft:.1f} ft" if em else "N/A"
        h3_val = f"{h3m.vertical_rmse_ft:.1f} ft" if h3m else "N/A"
        hs_val = f"{hsm.vertical_rmse_ft:.1f} ft" if hsm else "N/A"
        lines.append(f"{f'Vertical RMSE @ {hz}s':<{col1_w}} | {e_val:<{col2_w}} | {h3_val:<{col3_w}} | {hs_val:<{col4_w}}")

    lines.append(sep_line)

    # Classification metrics
    lines.append(f"{'Total Samples (pair-windows)':<{col1_w}} | {easy_conf.total_samples:<{col2_w}} | {hard_3step_conf.total_samples:<{col3_w}} | {hard_smoothed_conf.total_samples:<{col4_w}}")
    lines.append(f"{'Positive Samples':<{col1_w}} | {easy_conf.positive_samples:<{col2_w}} | {hard_3step_conf.positive_samples:<{col3_w}} | {hard_smoothed_conf.positive_samples:<{col4_w}}")
    lines.append(f"{'Negative Samples':<{col1_w}} | {easy_conf.negative_samples:<{col2_w}} | {hard_3step_conf.negative_samples:<{col3_w}} | {hard_smoothed_conf.negative_samples:<{col4_w}}")
    lines.append(f"{'True Positives (TP)':<{col1_w}} | {easy_conf.true_positives:<{col2_w}} | {hard_3step_conf.true_positives:<{col3_w}} | {hard_smoothed_conf.true_positives:<{col4_w}}")
    lines.append(f"{'False Positives (FP)':<{col1_w}} | {easy_conf.false_positives:<{col2_w}} | {hard_3step_conf.false_positives:<{col3_w}} | {hard_smoothed_conf.false_positives:<{col4_w}}")
    lines.append(f"{'False Negatives (FN)':<{col1_w}} | {easy_conf.false_negatives:<{col2_w}} | {hard_3step_conf.false_negatives:<{col3_w}} | {hard_smoothed_conf.false_negatives:<{col4_w}}")
    lines.append(f"{'True Negatives (TN)':<{col1_w}} | {easy_conf.true_negatives:<{col2_w}} | {hard_3step_conf.true_negatives:<{col3_w}} | {hard_smoothed_conf.true_negatives:<{col4_w}}")
    lines.append(f"{'Precision':<{col1_w}} | {easy_conf.precision:<{col2_w}.4f} | {hard_3step_conf.precision:<{col3_w}.4f} | {hard_smoothed_conf.precision:<{col4_w}.4f}")
    lines.append(f"{'Recall':<{col1_w}} | {easy_conf.recall:<{col2_w}.4f} | {hard_3step_conf.recall:<{col3_w}.4f} | {hard_smoothed_conf.recall:<{col4_w}.4f}")
    lines.append(f"{'F1 Score':<{col1_w}} | {easy_conf.f1_score:<{col2_w}.4f} | {hard_3step_conf.f1_score:<{col3_w}.4f} | {hard_smoothed_conf.f1_score:<{col4_w}.4f}")
    lines.append(f"{'False Alarms / 1000 Negatives':<{col1_w}} | {easy_conf.false_alarms_per_1000_negatives:<{col2_w}.2f} | {hard_3step_conf.false_alarms_per_1000_negatives:<{col3_w}.2f} | {hard_smoothed_conf.false_alarms_per_1000_negatives:<{col4_w}.2f}")

    lines.append(sep_line)

    # Lead-time distribution & TTC MAE
    e_lt_min = f"{easy_conf.tp_lead_time_min_s:.1f} s" if easy_conf.tp_lead_time_min_s is not None else "N/A"
    h3_lt_min = f"{hard_3step_conf.tp_lead_time_min_s:.1f} s" if hard_3step_conf.tp_lead_time_min_s is not None else "N/A"
    hs_lt_min = f"{hard_smoothed_conf.tp_lead_time_min_s:.1f} s" if hard_smoothed_conf.tp_lead_time_min_s is not None else "N/A"
    lines.append(f"{'TP Lead Time: Min':<{col1_w}} | {e_lt_min:<{col2_w}} | {h3_lt_min:<{col3_w}} | {hs_lt_min:<{col4_w}}")

    e_lt_med = f"{easy_conf.tp_lead_time_median_s:.1f} s" if easy_conf.tp_lead_time_median_s is not None else "N/A"
    h3_lt_med = f"{hard_3step_conf.tp_lead_time_median_s:.1f} s" if hard_3step_conf.tp_lead_time_median_s is not None else "N/A"
    hs_lt_med = f"{hard_smoothed_conf.tp_lead_time_median_s:.1f} s" if hard_smoothed_conf.tp_lead_time_median_s is not None else "N/A"
    lines.append(f"{'TP Lead Time: Median':<{col1_w}} | {e_lt_med:<{col2_w}} | {h3_lt_med:<{col3_w}} | {hs_lt_med:<{col4_w}}")

    e_lt_max = f"{easy_conf.tp_lead_time_max_s:.1f} s" if easy_conf.tp_lead_time_max_s is not None else "N/A"
    h3_lt_max = f"{hard_3step_conf.tp_lead_time_max_s:.1f} s" if hard_3step_conf.tp_lead_time_max_s is not None else "N/A"
    hs_lt_max = f"{hard_smoothed_conf.tp_lead_time_max_s:.1f} s" if hard_smoothed_conf.tp_lead_time_max_s is not None else "N/A"
    lines.append(f"{'TP Lead Time: Max':<{col1_w}} | {e_lt_max:<{col2_w}} | {h3_lt_max:<{col3_w}} | {hs_lt_max:<{col4_w}}")

    e_ttc = f"{easy_conf.tp_time_to_conflict_mae_s:.2f} s" if easy_conf.tp_time_to_conflict_mae_s is not None else "N/A"
    h3_ttc = f"{hard_3step_conf.tp_time_to_conflict_mae_s:.2f} s" if hard_3step_conf.tp_time_to_conflict_mae_s is not None else "N/A"
    hs_ttc = f"{hard_smoothed_conf.tp_time_to_conflict_mae_s:.2f} s" if hard_smoothed_conf.tp_time_to_conflict_mae_s is not None else "N/A"
    lines.append(f"{'TP Time-to-Conflict MAE':<{col1_w}} | {e_ttc:<{col2_w}} | {h3_ttc:<{col3_w}} | {hs_ttc:<{col4_w}}")

    # TTC Buckets recall
    lines.append(sep_line)
    lines.append("Recall by True TTC Bucket:")
    for b_key in ("0-60", "60-120", "120-180", "180-240", "240-300"):
        e_b = easy_conf.ttc_buckets.get(b_key, {}) if easy_conf.ttc_buckets else {}
        h3_b = hard_3step_conf.ttc_buckets.get(b_key, {}) if hard_3step_conf.ttc_buckets else {}
        hs_b = hard_smoothed_conf.ttc_buckets.get(b_key, {}) if hard_smoothed_conf.ttc_buckets else {}

        e_str = f"{e_b.get('recall', 0.0):.4f} ({e_b.get('tp', 0)}/{e_b.get('pos', 0)})" if e_b else "N/A"
        h3_str = f"{h3_b.get('recall', 0.0):.4f} ({h3_b.get('tp', 0)}/{h3_b.get('pos', 0)})" if h3_b else "N/A"
        hs_str = f"{hs_b.get('recall', 0.0):.4f} ({hs_b.get('tp', 0)}/{hs_b.get('pos', 0)})" if hs_b else "N/A"
        lines.append(f"{f'  - TTC {b_key}s':<{col1_w}} | {e_str:<{col2_w}} | {h3_str:<{col3_w}} | {hs_str:<{col4_w}}")

    # Subgroup metrics
    if hard_3step_conf.subgroup_metrics and hard_smoothed_conf.subgroup_metrics:
        lines.append(sep_line)
        lines.append("Performance by Conflict Scenario Type:")
        h3_m = hard_3step_conf.subgroup_metrics.get("maneuvering", {})
        hs_m = hard_smoothed_conf.subgroup_metrics.get("maneuvering", {})
        h3_m_rec = f"{h3_m.get('recall', 0.0):.4f}"
        hs_m_rec = f"{hs_m.get('recall', 0.0):.4f}"
        lines.append(
            f"{'  - Maneuvering Recall':<{col1_w}} | {'N/A (clean)':<{col2_w}} | "
            f"{h3_m_rec:<{col3_w}} | {hs_m_rec:<{col4_w}}"
        )
        h3_m_prec = f"{h3_m.get('precision', 0.0):.4f}"
        hs_m_prec = f"{hs_m.get('precision', 0.0):.4f}"
        lines.append(
            f"{'  - Maneuvering Precision':<{col1_w}} | {'N/A (clean)':<{col2_w}} | "
            f"{h3_m_prec:<{col3_w}} | {hs_m_prec:<{col4_w}}"
        )

        h3_nm = hard_3step_conf.subgroup_metrics.get("non_maneuvering", {})
        hs_nm = hard_smoothed_conf.subgroup_metrics.get("non_maneuvering", {})
        h3_nm_rec = f"{h3_nm.get('recall', 0.0):.4f}"
        hs_nm_rec = f"{hs_nm.get('recall', 0.0):.4f}"
        e_rec = f"{easy_conf.recall:.4f}"
        lines.append(
            f"{'  - Non-Maneuvering Recall':<{col1_w}} | {e_rec:<{col2_w}} | "
            f"{h3_nm_rec:<{col3_w}} | {hs_nm_rec:<{col4_w}}"
        )
        h3_nm_prec = f"{h3_nm.get('precision', 0.0):.4f}"
        hs_nm_prec = f"{hs_nm.get('precision', 0.0):.4f}"
        e_prec = f"{easy_conf.precision:.4f}"
        lines.append(
            f"{'  - Non-Maneuvering Precision':<{col1_w}} | {e_prec:<{col2_w}} | "
            f"{h3_nm_prec:<{col3_w}} | {hs_nm_prec:<{col4_w}}"
        )

    lines.append("=" * len(header))
    return "\n".join(lines)


def format_hard_comparison_table(
    h1_3step_metrics: list[PositionErrorMetric],
    h1_3step_conf: ConflictEvaluationResult,
    h1_smoothed_metrics: list[PositionErrorMetric],
    h1_smoothed_conf: ConflictEvaluationResult,
    hl_3step_metrics: list[PositionErrorMetric],
    hl_3step_conf: ConflictEvaluationResult,
    hl_smoothed_metrics: list[PositionErrorMetric],
    hl_smoothed_conf: ConflictEvaluationResult,
) -> str:
    """Format a 4-column side-by-side comparison table between hard_v1 and hard_large baselines."""
    h1_3_map = {m.horizon_s: m for m in h1_3step_metrics}
    h1_s_map = {m.horizon_s: m for m in h1_smoothed_metrics}
    hl_3_map = {m.horizon_s: m for m in hl_3step_metrics}
    hl_s_map = {m.horizon_s: m for m in hl_smoothed_metrics}

    col1_w = 34
    col2_w = 26
    col3_w = 26
    col4_w = 26
    col5_w = 26

    header = (
        f"{'Metric':<{col1_w}} | {'cv_3step (hard_v1)':<{col2_w}} | "
        f"{'cv_smoothed (hard_v1)':<{col3_w}} | {'cv_3step (hard_large)':<{col4_w}} | "
        f"{'cv_smoothed (hard_large)':<{col5_w}}"
    )
    sep_line = (
        "-" * col1_w + "-+-" + "-" * col2_w + "-+-" + "-" * col3_w + "-+-" +
        "-" * col4_w + "-+-" + "-" * col5_w
    )

    lines = [
        "=" * len(header),
        "SIDE-BY-SIDE BENCHMARK COMPARISON: hard_v1 vs hard_large (TEST SPLIT, 95% CIs)",
        "=" * len(header),
        header,
        sep_line,
    ]

    # Scenario & GT conflict counts
    lines.append(
        f"{'Test Scenarios Evaluated':<{col1_w}} | "
        f"{h1_3step_conf.num_scenarios:<{col2_w}} | {h1_smoothed_conf.num_scenarios:<{col3_w}} | "
        f"{hl_3step_conf.num_scenarios:<{col4_w}} | {hl_smoothed_conf.num_scenarios:<{col5_w}}"
    )
    lines.append(
        f"{'Distinct GT Conflict Pairs':<{col1_w}} | "
        f"{h1_3step_conf.num_gt_conflict_pairs:<{col2_w}} | {h1_smoothed_conf.num_gt_conflict_pairs:<{col3_w}} | "
        f"{hl_3step_conf.num_gt_conflict_pairs:<{col4_w}} | {hl_smoothed_conf.num_gt_conflict_pairs:<{col5_w}}"
    )

    lines.append(sep_line)

    # Horizontal RMSE
    for hz in (30, 60, 120, 180, 240, 300):
        v1_3 = f"{h1_3_map[hz].horizontal_rmse_nm:.4f} NM" if hz in h1_3_map else "N/A"
        v1_s = f"{h1_s_map[hz].horizontal_rmse_nm:.4f} NM" if hz in h1_s_map else "N/A"
        vl_3 = f"{hl_3_map[hz].horizontal_rmse_nm:.4f} NM" if hz in hl_3_map else "N/A"
        vl_s = f"{hl_s_map[hz].horizontal_rmse_nm:.4f} NM" if hz in hl_s_map else "N/A"
        lines.append(
            f"{f'Horizontal RMSE @ {hz}s':<{col1_w}} | {v1_3:<{col2_w}} | {v1_s:<{col3_w}} | "
            f"{vl_3:<{col4_w}} | {vl_s:<{col5_w}}"
        )

    lines.append(sep_line)

    # Vertical RMSE
    for hz in (30, 60, 120, 180, 240, 300):
        v1_3 = f"{h1_3_map[hz].vertical_rmse_ft:.1f} ft" if hz in h1_3_map else "N/A"
        v1_s = f"{h1_s_map[hz].vertical_rmse_ft:.1f} ft" if hz in h1_s_map else "N/A"
        vl_3 = f"{hl_3_map[hz].vertical_rmse_ft:.1f} ft" if hz in hl_3_map else "N/A"
        vl_s = f"{hl_s_map[hz].vertical_rmse_ft:.1f} ft" if hz in hl_s_map else "N/A"
        lines.append(
            f"{f'Vertical RMSE @ {hz}s':<{col1_w}} | {v1_3:<{col2_w}} | {v1_s:<{col3_w}} | "
            f"{vl_3:<{col4_w}} | {vl_s:<{col5_w}}"
        )

    lines.append(sep_line)

    # Sample counts
    lines.append(
        f"{'Total Samples (pair-origins)':<{col1_w}} | "
        f"{h1_3step_conf.total_samples:<{col2_w}} | {h1_smoothed_conf.total_samples:<{col3_w}} | "
        f"{hl_3step_conf.total_samples:<{col4_w}} | {hl_smoothed_conf.total_samples:<{col5_w}}"
    )
    lines.append(
        f"{'Positive Samples (LoS)':<{col1_w}} | "
        f"{h1_3step_conf.positive_samples:<{col2_w}} | {h1_smoothed_conf.positive_samples:<{col3_w}} | "
        f"{hl_3step_conf.positive_samples:<{col4_w}} | {hl_smoothed_conf.positive_samples:<{col5_w}}"
    )
    lines.append(
        f"{'Negative Samples (Clean)':<{col1_w}} | "
        f"{h1_3step_conf.negative_samples:<{col2_w}} | {h1_smoothed_conf.negative_samples:<{col3_w}} | "
        f"{hl_3step_conf.negative_samples:<{col4_w}} | {hl_smoothed_conf.negative_samples:<{col5_w}}"
    )
    lines.append(
        f"{'True Positives (TP)':<{col1_w}} | "
        f"{h1_3step_conf.true_positives:<{col2_w}} | {h1_smoothed_conf.true_positives:<{col3_w}} | "
        f"{hl_3step_conf.true_positives:<{col4_w}} | {hl_smoothed_conf.true_positives:<{col5_w}}"
    )
    lines.append(
        f"{'False Positives (FP)':<{col1_w}} | "
        f"{h1_3step_conf.false_positives:<{col2_w}} | {h1_smoothed_conf.false_positives:<{col3_w}} | "
        f"{hl_3step_conf.false_positives:<{col4_w}} | {hl_smoothed_conf.false_positives:<{col5_w}}"
    )
    lines.append(
        f"{'False Negatives (FN)':<{col1_w}} | "
        f"{h1_3step_conf.false_negatives:<{col2_w}} | {h1_smoothed_conf.false_negatives:<{col3_w}} | "
        f"{hl_3step_conf.false_negatives:<{col4_w}} | {hl_smoothed_conf.false_negatives:<{col5_w}}"
    )
    lines.append(
        f"{'True Negatives (TN)':<{col1_w}} | "
        f"{h1_3step_conf.true_negatives:<{col2_w}} | {h1_smoothed_conf.true_negatives:<{col3_w}} | "
        f"{hl_3step_conf.true_negatives:<{col4_w}} | {hl_smoothed_conf.true_negatives:<{col5_w}}"
    )

    lines.append(sep_line)

    def _fmt_ci(val: float, ci_tup: tuple[float, float] | None) -> str:
        if ci_tup is None:
            return f"{val:.4f}"
        return f"{val:.4f} [{ci_tup[0]:.4f}, {ci_tup[1]:.4f}]"

    # Precision with CI
    lines.append(
        f"{'Precision [95% CI]':<{col1_w}} | "
        f"{_fmt_ci(h1_3step_conf.precision, h1_3step_conf.bootstrap_cis.precision if h1_3step_conf.bootstrap_cis else None):<{col2_w}} | "
        f"{_fmt_ci(h1_smoothed_conf.precision, h1_smoothed_conf.bootstrap_cis.precision if h1_smoothed_conf.bootstrap_cis else None):<{col3_w}} | "
        f"{_fmt_ci(hl_3step_conf.precision, hl_3step_conf.bootstrap_cis.precision if hl_3step_conf.bootstrap_cis else None):<{col4_w}} | "
        f"{_fmt_ci(hl_smoothed_conf.precision, hl_smoothed_conf.bootstrap_cis.precision if hl_smoothed_conf.bootstrap_cis else None):<{col5_w}}"
    )

    # Recall with CI
    lines.append(
        f"{'Recall [95% CI]':<{col1_w}} | "
        f"{_fmt_ci(h1_3step_conf.recall, h1_3step_conf.bootstrap_cis.recall if h1_3step_conf.bootstrap_cis else None):<{col2_w}} | "
        f"{_fmt_ci(h1_smoothed_conf.recall, h1_smoothed_conf.bootstrap_cis.recall if h1_smoothed_conf.bootstrap_cis else None):<{col3_w}} | "
        f"{_fmt_ci(hl_3step_conf.recall, hl_3step_conf.bootstrap_cis.recall if hl_3step_conf.bootstrap_cis else None):<{col4_w}} | "
        f"{_fmt_ci(hl_smoothed_conf.recall, hl_smoothed_conf.bootstrap_cis.recall if hl_smoothed_conf.bootstrap_cis else None):<{col5_w}}"
    )

    # F1 Score with CI
    lines.append(
        f"{'F1 Score [95% CI]':<{col1_w}} | "
        f"{_fmt_ci(h1_3step_conf.f1_score, h1_3step_conf.bootstrap_cis.f1 if h1_3step_conf.bootstrap_cis else None):<{col2_w}} | "
        f"{_fmt_ci(h1_smoothed_conf.f1_score, h1_smoothed_conf.bootstrap_cis.f1 if h1_smoothed_conf.bootstrap_cis else None):<{col3_w}} | "
        f"{_fmt_ci(hl_3step_conf.f1_score, hl_3step_conf.bootstrap_cis.f1 if hl_3step_conf.bootstrap_cis else None):<{col4_w}} | "
        f"{_fmt_ci(hl_smoothed_conf.f1_score, hl_smoothed_conf.bootstrap_cis.f1 if hl_smoothed_conf.bootstrap_cis else None):<{col5_w}}"
    )

    # Conflict Detection Rate with CI
    lines.append(
        f"{'Conflict Det Rate [95% CI]':<{col1_w}} | "
        f"{_fmt_ci(h1_3step_conf.conflict_detection_rate, h1_3step_conf.bootstrap_cis.conflict_detection_rate if h1_3step_conf.bootstrap_cis else None):<{col2_w}} | "
        f"{_fmt_ci(h1_smoothed_conf.conflict_detection_rate, h1_smoothed_conf.bootstrap_cis.conflict_detection_rate if h1_smoothed_conf.bootstrap_cis else None):<{col3_w}} | "
        f"{_fmt_ci(hl_3step_conf.conflict_detection_rate, hl_3step_conf.bootstrap_cis.conflict_detection_rate if hl_3step_conf.bootstrap_cis else None):<{col4_w}} | "
        f"{_fmt_ci(hl_smoothed_conf.conflict_detection_rate, hl_smoothed_conf.bootstrap_cis.conflict_detection_rate if hl_smoothed_conf.bootstrap_cis else None):<{col5_w}}"
    )

    lines.append(sep_line)

    def _fmt_opt(val: float | None, unit: str = " s", prec: int = 1) -> str:
        return f"{val:.{prec}f}{unit}" if val is not None else "N/A"

    # First-detection lead time
    lines.append(
        f"{'Conflict 1st Det Lead: Min':<{col1_w}} | "
        f"{_fmt_opt(h1_3step_conf.first_detection_lead_min_s):<{col2_w}} | "
        f"{_fmt_opt(h1_smoothed_conf.first_detection_lead_min_s):<{col3_w}} | "
        f"{_fmt_opt(hl_3step_conf.first_detection_lead_min_s):<{col4_w}} | "
        f"{_fmt_opt(hl_smoothed_conf.first_detection_lead_min_s):<{col5_w}}"
    )
    lines.append(
        f"{'Conflict 1st Det Lead: Median':<{col1_w}} | "
        f"{_fmt_opt(h1_3step_conf.first_detection_lead_median_s):<{col2_w}} | "
        f"{_fmt_opt(h1_smoothed_conf.first_detection_lead_median_s):<{col3_w}} | "
        f"{_fmt_opt(hl_3step_conf.first_detection_lead_median_s):<{col4_w}} | "
        f"{_fmt_opt(hl_smoothed_conf.first_detection_lead_median_s):<{col5_w}}"
    )
    lines.append(
        f"{'Conflict 1st Det Lead: Max':<{col1_w}} | "
        f"{_fmt_opt(h1_3step_conf.first_detection_lead_max_s):<{col2_w}} | "
        f"{_fmt_opt(h1_smoothed_conf.first_detection_lead_max_s):<{col3_w}} | "
        f"{_fmt_opt(hl_3step_conf.first_detection_lead_max_s):<{col4_w}} | "
        f"{_fmt_opt(hl_smoothed_conf.first_detection_lead_max_s):<{col5_w}}"
    )

    # Window-level metrics
    lines.append(
        f"{'TP Lead Time: Median':<{col1_w}} | "
        f"{_fmt_opt(h1_3step_conf.tp_lead_time_median_s):<{col2_w}} | "
        f"{_fmt_opt(h1_smoothed_conf.tp_lead_time_median_s):<{col3_w}} | "
        f"{_fmt_opt(hl_3step_conf.tp_lead_time_median_s):<{col4_w}} | "
        f"{_fmt_opt(hl_smoothed_conf.tp_lead_time_median_s):<{col5_w}}"
    )
    lines.append(
        f"{'TP Time-to-Conflict MAE':<{col1_w}} | "
        f"{_fmt_opt(h1_3step_conf.tp_time_to_conflict_mae_s, prec=2):<{col2_w}} | "
        f"{_fmt_opt(h1_smoothed_conf.tp_time_to_conflict_mae_s, prec=2):<{col3_w}} | "
        f"{_fmt_opt(hl_3step_conf.tp_time_to_conflict_mae_s, prec=2):<{col4_w}} | "
        f"{_fmt_opt(hl_smoothed_conf.tp_time_to_conflict_mae_s, prec=2):<{col5_w}}"
    )
    lines.append(
        f"{'False Alarms / 1000 Negatives':<{col1_w}} | "
        f"{f'{h1_3step_conf.false_alarms_per_1000_negatives:.2f}':<{col2_w}} | "
        f"{f'{h1_smoothed_conf.false_alarms_per_1000_negatives:.2f}':<{col3_w}} | "
        f"{f'{hl_3step_conf.false_alarms_per_1000_negatives:.2f}':<{col4_w}} | "
        f"{f'{hl_smoothed_conf.false_alarms_per_1000_negatives:.2f}':<{col5_w}}"
    )

    lines.append(sep_line)

    # False Positive breakdown by category
    lines.append(
        f"{'FP: Injected Conflicts':<{col1_w}} | "
        f"{h1_3step_conf.fp_categories.get('injected_conflict', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_categories.get('injected_conflict', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_categories.get('injected_conflict', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_categories.get('injected_conflict', 0):<{col5_w}}"
    )
    lines.append(
        f"{'FP: Near-Miss Encounters':<{col1_w}} | "
        f"{h1_3step_conf.fp_categories.get('near_miss', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_categories.get('near_miss', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_categories.get('near_miss', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_categories.get('near_miss', 0):<{col5_w}}"
    )
    lines.append(
        f"{'FP: Other Clean Encounters':<{col1_w}} | "
        f"{h1_3step_conf.fp_categories.get('other_clean', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_categories.get('other_clean', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_categories.get('other_clean', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_categories.get('other_clean', 0):<{col5_w}}"
    )

    # False Positive breakdown by cause
    lines.append(
        f"{'FP Cause: Lateral Error':<{col1_w}} | "
        f"{h1_3step_conf.fp_causes.get('lateral', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_causes.get('lateral', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_causes.get('lateral', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_causes.get('lateral', 0):<{col5_w}}"
    )
    lines.append(
        f"{'FP Cause: Vertical Error':<{col1_w}} | "
        f"{h1_3step_conf.fp_causes.get('vertical', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_causes.get('vertical', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_causes.get('vertical', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_causes.get('vertical', 0):<{col5_w}}"
    )
    lines.append(
        f"{'FP Cause: Both Errors':<{col1_w}} | "
        f"{h1_3step_conf.fp_causes.get('both', 0):<{col2_w}} | "
        f"{h1_smoothed_conf.fp_causes.get('both', 0):<{col3_w}} | "
        f"{hl_3step_conf.fp_causes.get('both', 0):<{col4_w}} | "
        f"{hl_smoothed_conf.fp_causes.get('both', 0):<{col5_w}}"
    )

    lines.append(sep_line)

    # Subset metrics
    for sub, name in [
        ("maneuvering", "Maneuvering Conflict"),
        ("non_maneuvering", "Non-Maneuvering Conflict"),
        ("near_miss", "Near-Miss Encounter"),
        ("other_clean", "Other Clean Encounter"),
    ]:
        def _fmt_sub(conf: ConflictEvaluationResult, k: str) -> str:
            sm = conf.subset_metrics.get(k, {})
            r = sm.get("recall", 0.0)
            p = sm.get("precision", 0.0)
            return f"rec={r:.4f}, prec={p:.4f}"

        lines.append(
            f"{name:<{col1_w}} | "
            f"{_fmt_sub(h1_3step_conf, sub):<{col2_w}} | "
            f"{_fmt_sub(h1_smoothed_conf, sub):<{col3_w}} | "
            f"{_fmt_sub(hl_3step_conf, sub):<{col4_w}} | "
            f"{_fmt_sub(hl_smoothed_conf, sub):<{col5_w}}"
        )

    lines.append("=" * len(header))
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for run_baseline."""
    parser = argparse.ArgumentParser(
        description="Run trajectory baseline and conflict evaluation harness on test split."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="hard_large",
        choices=["easy", "hard", "hard_large"],
        help="Dataset variant to evaluate (default: hard_large)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cv_3step",
        choices=["cv_3step", "cv_smoothed", "lstm"],
        help="Predictor baseline model (default: cv_3step)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for --model lstm",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run cv_3step (easy), cv_3step (hard), and cv_smoothed (hard) and print side-by-side comparison table",
    )
    parser.add_argument(
        "--compare-hard",
        action="store_true",
        help="Run cv_3step and cv_smoothed on BOTH hard_v1 and hard_large and print 4-way comparison table",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to synthetic scenario directory (overrides default for variant)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration",
    )
    args = parser.parse_args()

    if args.compare_hard:
        print("\n>>> EVALUATING 1/4: cv_3step on hard_v1 TEST...")
        h1_3_metrics, h1_3_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard",
            model_name="cv_3step",
        )
        print("\n>>> EVALUATING 2/4: cv_smoothed on hard_v1 TEST...")
        h1_s_metrics, h1_s_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard",
            model_name="cv_smoothed",
        )
        print("\n>>> EVALUATING 3/4: cv_3step on hard_large TEST...")
        hl_3_metrics, hl_3_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard_large",
            model_name="cv_3step",
        )
        print("\n>>> EVALUATING 4/4: cv_smoothed on hard_large TEST...")
        hl_s_metrics, hl_s_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard_large",
            model_name="cv_smoothed",
        )

        table_str = format_hard_comparison_table(
            h1_3step_metrics=h1_3_metrics,
            h1_3step_conf=h1_3_conf,
            h1_smoothed_metrics=h1_s_metrics,
            h1_smoothed_conf=h1_s_conf,
            hl_3step_metrics=hl_3_metrics,
            hl_3step_conf=hl_3_conf,
            hl_smoothed_metrics=hl_s_metrics,
            hl_smoothed_conf=hl_s_conf,
        )
        print("\n\n" + table_str)

    elif args.compare:
        easy_metrics, easy_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="easy",
            model_name="cv_3step",
        )
        print("\n\n")
        hard_3step_metrics, hard_3step_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard",
            model_name="cv_3step",
        )
        print("\n\n")
        hard_smoothed_metrics, hard_smoothed_conf = run_test_evaluation(
            data_dir=None,
            config_path=args.config,
            variant="hard",
            model_name="cv_smoothed",
        )
        print("\n\n")
        print(format_comparison_table(
            easy_metrics=easy_metrics,
            easy_conf=easy_conf,
            hard_3step_metrics=hard_3step_metrics,
            hard_3step_conf=hard_3step_conf,
            hard_smoothed_metrics=hard_smoothed_metrics,
            hard_smoothed_conf=hard_smoothed_conf,
        ))
    else:
        run_test_evaluation(
            data_dir=args.data_dir,
            config_path=args.config,
            variant=args.variant,
            model_name=args.model,
            checkpoint=args.checkpoint,
            split=args.split,
        )


if __name__ == "__main__":
    main()


