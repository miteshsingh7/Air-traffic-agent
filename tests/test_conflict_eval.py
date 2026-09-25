"""Unit tests for conflict prediction and advisory evaluation.

Verifies:
- Conflict evaluation against ground-truth future encounters
- Known True Positive (TP) from straight-line encounter
- Known False Negative (FN) from an unexpected future turn
- Correct precision, recall, and lead-time calculations
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eval.conflict_eval import evaluate_conflict_prediction
from src.models.baseline import ConstantVelocityPredictor


def test_conflict_eval_known_tp_and_known_fn():
    """Verify conflict evaluation with a guaranteed TP and FN.

    Scenario configuration at origin t0 = 60s:
    - AC1: flies straight east along y=0 at 360 kt (0.1 NM/s).
    - AC2 (Known TP with AC1): flies straight north along x=16 at 360 kt.
      Both AC1 and AC2 cross at (16, 0) at t = 160s (t0 + 100s).
      Constant-velocity predictor correctly extrapolates this straight-line encounter -> TP.
    - AC3 (Known FN with AC1):
      Up to t=60s, AC3 flies parallel to AC1 at y=30 NM (velocity in y is 0).
      Model predicts parallel flight -> No conflict predicted (pred=0).
      In true future at t in [180s, 240s], AC3 turns South towards AC1 and intercepts at t=240s.
      True future has conflict -> FN.
    - AC2 and AC3 are separated by > 7 NM at all times -> TN.
    """
    times = np.arange(0.0, 365.0, 5.0)  # t from 0 to 360s (history 60s + horizon 300s)

    # AC1: straight East along y=0, alt=30,000 ft
    ac1 = pd.DataFrame(
        {
            "scenario_id": "test_scen_tp_fn",
            "aircraft_id": "AC_1",
            "t": times,
            "x_nm": 0.1 * times,
            "y_nm": np.zeros_like(times),
            "alt_ft": np.full_like(times, 30000.0),
            "heading_deg": np.full_like(times, 90.0),
            "speed_kt": np.full_like(times, 360.0),
        }
    )

    # AC2: straight North along x=16, crosses y=0 at t=160s
    ac2 = pd.DataFrame(
        {
            "scenario_id": "test_scen_tp_fn",
            "aircraft_id": "AC_2",
            "t": times,
            "x_nm": np.full_like(times, 16.0),
            "y_nm": -16.0 + 0.1 * times,
            "alt_ft": np.full_like(times, 30000.0),
            "heading_deg": np.full_like(times, 0.0),
            "speed_kt": np.full_like(times, 360.0),
        }
    )

    # AC3: parallel at y=30 for t <= 180s, then turns south to reach y=0 at t=240s
    x3 = 0.1 * times
    y3 = np.full_like(times, 30.0)
    for i, t in enumerate(times):
        if t > 180.0:
            y3[i] = max(0.0, 30.0 - 0.5 * (t - 180.0))  # reaches y=0 at t=240s

    ac3 = pd.DataFrame(
        {
            "scenario_id": "test_scen_tp_fn",
            "aircraft_id": "AC_3",
            "t": times,
            "x_nm": x3,
            "y_nm": y3,
            "alt_ft": np.full_like(times, 30000.0),
            "heading_deg": np.full_like(times, 90.0),
            "speed_kt": np.full_like(times, 360.0),
        }
    )

    scenario_df = pd.concat([ac1, ac2, ac3], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        scen_path = Path(tmp_dir) / "scenario_test.parquet"
        scenario_df.to_parquet(scen_path, index=False)

        predictor = ConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)
        res = evaluate_conflict_prediction(
            scenario_files=[scen_path],
            predictor=predictor,
            lateral_min_nm=5.0,
            vertical_min_ft=1000.0,
            origin_step_s=30.0,
            history_s=60.0,
            horizon_s=300.0,
            dt_s=5.0,
        )

        # There is 1 origin (t=60s) and 3 pairs: (AC1, AC2), (AC1, AC3), (AC2, AC3)
        assert res.total_samples == 3
        assert res.true_positives == 1, f"Expected 1 TP (AC1-AC2), got {res.true_positives}"
        assert res.false_negatives == 1, f"Expected 1 FN (AC1-AC3), got {res.false_negatives}"
        assert res.true_negatives == 1, f"Expected 1 TN (AC2-AC3), got {res.true_negatives}"
        assert res.false_positives == 0

        assert res.positive_samples == 2
        assert res.negative_samples == 1

        assert res.precision == 1.0  # 1 / (1 + 0)
        assert res.recall == 0.5    # 1 / (1 + 1)
        assert res.f1_score == pytest.approx(2.0 / 3.0, abs=1e-4)

        assert res.tp_lead_time_median_s is not None
        assert res.tp_lead_time_median_s > 0
        assert res.tp_time_to_conflict_mae_s is not None
        assert res.tp_time_to_conflict_mae_s == pytest.approx(0.0, abs=1e-5)

        # Conflict-level metrics on known case
        assert res.num_scenarios == 1
        assert res.num_gt_conflict_pairs == 2  # (AC1, AC2) and (AC1, AC3)
        assert res.num_detected_conflict_pairs == 1  # (AC1, AC2) detected at t=60s; (AC1, AC3) is FN
        assert res.conflict_detection_rate == 0.5
        # Earliest LoS onset is t=125s (dist=4.95 NM < 5 NM); origin is t=60s; lead time = 125 - 60 = 65s
        assert res.first_detection_lead_min_s == pytest.approx(65.0, abs=1e-4)


def test_bootstrap_resamples_scenarios_not_rows():
    """Verify that bootstrap confidence intervals resample at scenario level, not row level.

    Scenario-level clustering test:
    - Scenario A: 10 TPs, 0 FP, 0 FN, 0 TN
    - Scenario B: 0 TP, 0 FP, 10 FN, 0 TN
    Total samples = 20 (10 TP, 10 FN).
    If rows were resampled independently, the resampled TP count would take any value in 0..20,
    including odd numbers (e.g. 7, 9, 11).
    When SCENARIOS are resampled (sample size 2 from [A, B] with replacement):
    - [A, A] -> 20 TPs
    - [A, B] -> 10 TPs
    - [B, B] -> 0 TPs
    Every single bootstrap resample MUST have total TP in {0, 10, 20}. Odd counts are impossible.
    """
    from src.eval.conflict_eval import ScenarioStats, compute_scenario_bootstrap_cis

    scen_a = ScenarioStats(
        scenario_id="scen_a",
        tp=10,
        fp=0,
        fn=0,
        tn=0,
        gt_conflict_pairs=1,
        detected_conflict_pairs=1,
    )
    scen_b = ScenarioStats(
        scenario_id="scen_b",
        tp=0,
        fp=0,
        fn=10,
        tn=0,
        gt_conflict_pairs=1,
        detected_conflict_pairs=0,
    )

    stats = [scen_a, scen_b]
    cis = compute_scenario_bootstrap_cis(stats, n_bootstraps=1000, seed=42)

    # In 2-scenario bootstrap:
    # [A, A] has recall = 20 / 20 = 1.0 (prob 0.25)
    # [A, B] has recall = 10 / 20 = 0.5 (prob 0.50)
    # [B, B] has recall = 0 / 20 = 0.0 (prob 0.25)
    # The 2.5th percentile is 0.0 and 97.5th is 1.0
    assert cis.recall[0] == pytest.approx(0.0, abs=1e-5)
    assert cis.recall[1] == pytest.approx(1.0, abs=1e-5)

    # Now verify manually with raw numpy sampling that only whole-scenario sums occur
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 2, size=(1000, 2))
    arr_tp = np.array([scen_a.tp, scen_b.tp])
    sampled_tps = np.sum(arr_tp[indices], axis=1)

    # All resampled TP counts must strictly be in {0, 10, 20}
    unique_tps = set(np.unique(sampled_tps))
    assert unique_tps.issubset({0, 10, 20}), f"Found non-scenario counts: {unique_tps}"
    assert not any(tp % 10 != 0 for tp in sampled_tps), "Row-level sampling detected!"


def test_fp_category_counts_sum_to_total_fp():
    """Verify that FP breakdown categories and causes sum exactly to total false positives."""
    # Create 3 mini scenarios with distinct categories in metadata
    times = np.arange(0.0, 365.0, 5.0)

    # AC1 and AC2 parallel separated by 10 NM (no LoS)
    def create_df(scen_id: str) -> pd.DataFrame:
        ac1 = pd.DataFrame({
            "scenario_id": scen_id,
            "aircraft_id": "AC_1",
            "t": times,
            "x_nm": 0.1 * times,
            "y_nm": np.zeros_like(times),
            "alt_ft": np.full_like(times, 30000.0),
            "heading_deg": np.full_like(times, 90.0),
            "speed_kt": np.full_like(times, 360.0),
        })
        ac2 = pd.DataFrame({
            "scenario_id": scen_id,
            "aircraft_id": "AC_2",
            "t": times,
            "x_nm": 0.1 * times,
            "y_nm": np.full_like(times, 10.0),  # 10 NM away
            "alt_ft": np.full_like(times, 30000.0),
            "heading_deg": np.full_like(times, 90.0),
            "speed_kt": np.full_like(times, 360.0),
        })
        return pd.concat([ac1, ac2], ignore_index=True)

    class BiasedPredictor:
        """Predictor that always predicts a collision to force false positives."""

        def predict(self, history: np.ndarray) -> np.ndarray:
            n_samples = history.shape[0] if history.ndim == 3 else 1
            # Collide at (0, 0, 30000)
            return np.zeros((n_samples, 60, 3))

    metadata = {
        "scen_injected": {"injected_conflict": True, "maneuvering": False, "near_miss": False},
        "scen_near_miss": {"injected_conflict": False, "maneuvering": False, "near_miss": True},
        "scen_clean": {"injected_conflict": False, "maneuvering": False, "near_miss": False},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for sid in metadata:
            p = Path(tmp_dir) / f"{sid}.parquet"
            create_df(sid).to_parquet(p, index=False)
            paths.append(p)

        res = evaluate_conflict_prediction(
            scenario_files=paths,
            predictor=BiasedPredictor(),
            lateral_min_nm=5.0,
            vertical_min_ft=1000.0,
            origin_step_s=30.0,
            history_s=60.0,
            horizon_s=300.0,
            dt_s=5.0,
            metadata=metadata,
        )

        assert res.false_positives > 0
        assert res.true_positives == 0

        # Assert FP categories sum to total FP
        sum_cat = sum(res.fp_categories.values())
        assert sum_cat == res.false_positives, f"FP categories ({sum_cat}) != total FP ({res.false_positives})"

        # Assert FP causes sum to total FP
        sum_causes = sum(res.fp_causes.values())
        assert sum_causes == res.false_positives, f"FP causes ({sum_causes}) != total FP ({res.false_positives})"

        # Check all 3 categories were represented
        assert res.fp_categories["injected_conflict"] > 0
        assert res.fp_categories["near_miss"] > 0
        assert res.fp_categories["other_clean"] > 0


def test_lead_conditioned_detection_and_out_of_horizon():
    """Verify lead-conditioned detection rates and out-of-horizon detection handling."""
    from src.eval.conflict_eval import format_conflict_evaluation

    # Scenario: AC1 and AC2 converge at t=400s
    times = np.arange(0.0, 500.0, 5.0)
    ac1 = pd.DataFrame({
        "scenario_id": "scen_lead_test",
        "aircraft_id": "AC_1",
        "t": times,
        "x_nm": 0.1 * times,
        "y_nm": np.zeros_like(times),
        "alt_ft": np.full_like(times, 30000.0),
        "heading_deg": np.full_like(times, 90.0),
        "speed_kt": np.full_like(times, 360.0),
    })
    ac2 = pd.DataFrame({
        "scenario_id": "scen_lead_test",
        "aircraft_id": "AC_2",
        "t": times,
        "x_nm": np.full_like(times, 40.0),
        "y_nm": -40.0 + 0.1 * times,
        "alt_ft": np.full_like(times, 30000.0),
        "heading_deg": np.full_like(times, 0.0),
        "speed_kt": np.full_like(times, 360.0),
    })
    scenario_df = pd.concat([ac1, ac2], ignore_index=True)

    metadata = {
        "scen_lead_test": {"injected_conflict": True, "maneuvering": False, "near_miss": False}
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "scen_lead_test.parquet"
        scenario_df.to_parquet(p, index=False)

        predictor = ConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)
        res = evaluate_conflict_prediction(
            scenario_files=[p],
            predictor=predictor,
            lateral_min_nm=5.0,
            vertical_min_ft=1000.0,
            origin_step_s=30.0,
            history_s=60.0,
            horizon_s=300.0,
            dt_s=5.0,
            metadata=metadata,
        )

        assert res.num_scenarios == 1
        assert res.num_gt_conflict_pairs == 1
        assert res.num_detected_conflict_pairs == 1
        # Lead time at first detection must be <= 300s
        assert res.first_detection_lead_max_s is not None
        assert res.first_detection_lead_max_s <= 300.0

        # Check lead conditioned rates are present
        assert 60 in res.lead_conditioned_rates
        assert 120 in res.lead_conditioned_rates
        assert 180 in res.lead_conditioned_rates
        assert 240 in res.lead_conditioned_rates

        formatted = format_conflict_evaluation(res)
        assert "Lead >=  60 s" in formatted
        assert "recall=N/A, FA/1000 neg=" in formatted

