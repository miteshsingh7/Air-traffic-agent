"""Tests for src/agent — ConflictAdvisoryAgent, types, and advisory pipeline.

Research simulation only — not for operational use.

Test Strategy
-------------
- Unit tests for frozen dataclass construction and properties.
- Unit tests for _compute_cpa and _classify with analytically-known geometry.
- Integration tests for advise_scenario with programmatically constructed
  2-aircraft scenarios (converging, parallel, vertically separated, clean).
- File-loading test (skipped when sample parquet absent).

All scenarios are constructed in-memory using numpy — no dependency on
generated dataset files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent import (
    ConflictAdvisory,
    ConflictAdvisoryAgent,
    PairwiseCPA,
    ScenarioAdvisoryReport,
)
from src.agent.advisor import TrajectoryPredictor
from src.conflict.risk import compute_conflict_risk
from src.models.baseline import (
    ConstantVelocityPredictor,
    SmoothedConstantVelocityPredictor,
)

# ---------------------------------------------------------------------------
# Helpers — tiny in-memory scenario builders
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _make_scenario_df(
    scenario_id: str,
    aircraft_tracks: dict[str, np.ndarray],
    dt_s: float = 5.0,
) -> pd.DataFrame:
    """Build a scenario DataFrame from dict of {acid: positions (T, 3)}.

    Positions columns: [x_nm, y_nm, alt_ft].
    """
    rows: list[dict] = []
    t_start = 0.0
    for acid, pos in aircraft_tracks.items():
        t_steps = t_start + np.arange(len(pos)) * dt_s
        for t, p in zip(t_steps, pos):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "aircraft_id": acid,
                    "t": float(t),
                    "x_nm": float(p[0]),
                    "y_nm": float(p[1]),
                    "alt_ft": float(p[2]),
                }
            )
    return pd.DataFrame(rows)


def _straight_track(
    x0: float,
    y0: float,
    z0: float,
    vx: float,
    vy: float,
    vz: float,
    n_steps: int = 200,
    dt_s: float = 5.0,
) -> np.ndarray:
    """Generate a straight constant-velocity trajectory."""
    t = np.arange(n_steps, dtype=np.float64) * dt_s
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# 1. Type construction tests
# ---------------------------------------------------------------------------


class TestPairwiseCPA:
    def test_construction_and_fields(self) -> None:
        cpa = PairwiseCPA(
            scenario_id="s1",
            aircraft_1="AC1",
            aircraft_2="AC2",
            origin_t=60.0,
            cpa_lateral_nm=3.0,
            cpa_vertical_ft=500.0,
            cpa_time_s=180.0,
            time_to_cpa_s=120.0,
            closure_rate_nm_per_s=0.1,
        )
        assert cpa.scenario_id == "s1"
        assert cpa.aircraft_1 == "AC1"
        assert cpa.cpa_lateral_nm == pytest.approx(3.0)
        assert cpa.time_to_cpa_s == pytest.approx(120.0)

    def test_is_frozen(self) -> None:
        cpa = PairwiseCPA(
            scenario_id="s1",
            aircraft_1="AC1",
            aircraft_2="AC2",
            origin_t=60.0,
            cpa_lateral_nm=3.0,
            cpa_vertical_ft=500.0,
            cpa_time_s=180.0,
            time_to_cpa_s=120.0,
            closure_rate_nm_per_s=0.1,
        )
        with pytest.raises(Exception):
            cpa.cpa_lateral_nm = 1.0  # type: ignore[misc]


class TestConflictAdvisory:
    def _make_cpa(self, lat: float = 2.0, vert: float = 300.0) -> PairwiseCPA:
        return PairwiseCPA(
            scenario_id="s1",
            aircraft_1="AC1",
            aircraft_2="AC2",
            origin_t=60.0,
            cpa_lateral_nm=lat,
            cpa_vertical_ft=vert,
            cpa_time_s=180.0,
            time_to_cpa_s=120.0,
            closure_rate_nm_per_s=0.05,
        )

    def test_flagged_advisory(self) -> None:
        adv = ConflictAdvisory(
            cpa=self._make_cpa(),
            is_conflict=True,
            p_risk=0.6,
            lead_time_s=90.0,
        )
        assert adv.is_conflict is True
        assert adv.lead_time_s == pytest.approx(90.0)

    def test_non_conflict_advisory(self) -> None:
        adv = ConflictAdvisory(
            cpa=self._make_cpa(lat=6.0, vert=2000.0),
            is_conflict=False,
            p_risk=0.0,
            lead_time_s=None,
        )
        assert adv.is_conflict is False
        assert adv.lead_time_s is None


class TestScenarioAdvisoryReport:
    def _make_report(self, flagged: bool) -> ScenarioAdvisoryReport:
        cpa = PairwiseCPA(
            scenario_id="s1",
            aircraft_1="AC1",
            aircraft_2="AC2",
            origin_t=60.0,
            cpa_lateral_nm=2.0 if flagged else 8.0,
            cpa_vertical_ft=400.0 if flagged else 3000.0,
            cpa_time_s=180.0,
            time_to_cpa_s=120.0,
            closure_rate_nm_per_s=0.05,
        )
        adv = ConflictAdvisory(
            cpa=cpa,
            is_conflict=flagged,
            p_risk=0.6 if flagged else 0.0,
            lead_time_s=90.0 if flagged else None,
        )
        pairs: frozenset[tuple[str, str]] = (
            frozenset({("AC1", "AC2")}) if flagged else frozenset()
        )
        return ScenarioAdvisoryReport(
            scenario_id="s1",
            num_aircraft=2,
            num_origins=5,
            advisories=(adv,),
            flagged_pairs=pairs,
            max_p_risk=0.6 if flagged else 0.0,
            earliest_lead_time_s=90.0 if flagged else None,
        )

    def test_has_conflict_true(self) -> None:
        report = self._make_report(flagged=True)
        assert report.has_conflict is True
        assert report.num_flagged == 1

    def test_has_conflict_false(self) -> None:
        report = self._make_report(flagged=False)
        assert report.has_conflict is False
        assert report.num_flagged == 0


# ---------------------------------------------------------------------------
# 2. ConflictAdvisoryAgent construction validation
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_valid_construction(self) -> None:
        pred = SmoothedConstantVelocityPredictor()
        agent = ConflictAdvisoryAgent(pred)
        assert agent.lateral_min_nm == pytest.approx(5.0)
        assert agent.vertical_min_ft == pytest.approx(1000.0)

    def test_invalid_predictor_raises(self) -> None:
        with pytest.raises(TypeError):
            ConflictAdvisoryAgent("not_a_predictor")  # type: ignore[arg-type]

    def test_invalid_lateral_min_raises(self) -> None:
        pred = SmoothedConstantVelocityPredictor()
        with pytest.raises(ValueError, match="lateral_min_nm"):
            ConflictAdvisoryAgent(pred, lateral_min_nm=0.0)

    def test_invalid_history_steps_raises(self) -> None:
        pred = SmoothedConstantVelocityPredictor()
        with pytest.raises(ValueError, match="history_steps"):
            ConflictAdvisoryAgent(pred, history_steps=2)

    def test_custom_thresholds(self) -> None:
        pred = SmoothedConstantVelocityPredictor()
        agent = ConflictAdvisoryAgent(pred, lateral_min_nm=3.0, vertical_min_ft=500.0)
        assert agent.lateral_min_nm == pytest.approx(3.0)
        assert agent.vertical_min_ft == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 3. _compute_cpa unit tests (analytical geometry)
# ---------------------------------------------------------------------------


class TestComputeCPA:
    """Test _compute_cpa with analytically verifiable geometries."""

    def _agent(self) -> ConflictAdvisoryAgent:
        return ConflictAdvisoryAgent(SmoothedConstantVelocityPredictor())

    def test_parallel_tracks_cpa_constant(self) -> None:
        """Two parallel aircraft 10 NM apart — CPA should equal initial separation."""
        agent = self._agent()
        # pred_1 at x=0, pred_2 at x=10, same y, same alt — no convergence
        H = 60
        pred_1 = np.zeros((H, 3))
        pred_1[:, 0] = 0.0  # x stays 0
        pred_2 = np.zeros((H, 3))
        pred_2[:, 0] = 10.0  # x stays 10
        pred_1[:, 2] = 20000.0
        pred_2[:, 2] = 20000.0

        cpa = agent._compute_cpa(pred_1, pred_2, 60.0, "s", "A", "B")
        assert cpa.cpa_lateral_nm == pytest.approx(10.0, abs=0.01)
        assert cpa.aircraft_1 == "A"
        assert cpa.aircraft_2 == "B"

    def test_converging_tracks_cpa_decreasing(self) -> None:
        """Two aircraft converging from 20 NM — CPA should be < 20 NM."""
        agent = self._agent()
        H = 60
        t = np.arange(H) * 5.0
        pred_1 = np.column_stack([np.zeros(H), np.zeros(H), np.full(H, 20000.0)])
        pred_2 = np.column_stack([20.0 - 0.05 * t, np.zeros(H), np.full(H, 20000.0)])

        cpa = agent._compute_cpa(pred_1, pred_2, 60.0, "s", "A", "B")
        assert cpa.cpa_lateral_nm < 20.0
        assert cpa.time_to_cpa_s > 0.0

    def test_cpa_time_consistent_with_origin(self) -> None:
        """cpa_time_s = origin_t + time_to_cpa_s."""
        agent = self._agent()
        H = 60
        pred_1 = np.zeros((H, 3))
        pred_2 = np.zeros((H, 3))
        pred_2[:, 0] = 8.0

        cpa = agent._compute_cpa(pred_1, pred_2, 90.0, "s", "A", "B")
        assert cpa.cpa_time_s == pytest.approx(cpa.origin_t + cpa.time_to_cpa_s)

    def test_closure_rate_positive_when_converging(self) -> None:
        """Closure rate should be positive for converging aircraft."""
        agent = self._agent()
        H = 60
        t = np.arange(H) * 5.0
        pred_1 = np.zeros((H, 3))
        pred_2 = np.column_stack([15.0 - 0.1 * t, np.zeros(H), np.zeros(H)])

        cpa = agent._compute_cpa(pred_1, pred_2, 60.0, "s", "A", "B")
        assert cpa.closure_rate_nm_per_s > 0.0


# ---------------------------------------------------------------------------
# 4. _classify unit tests (boundary values)
# ---------------------------------------------------------------------------


class TestClassify:
    def _agent(self) -> ConflictAdvisoryAgent:
        return ConflictAdvisoryAgent(
            SmoothedConstantVelocityPredictor(),
            lateral_min_nm=5.0,
            vertical_min_ft=1000.0,
        )

    def _cpa(self, lat: float, vert: float) -> PairwiseCPA:
        return PairwiseCPA(
            scenario_id="s",
            aircraft_1="A",
            aircraft_2="B",
            origin_t=60.0,
            cpa_lateral_nm=lat,
            cpa_vertical_ft=vert,
            cpa_time_s=180.0,
            time_to_cpa_s=120.0,
            closure_rate_nm_per_s=0.0,
        )

    def test_both_axes_violated_is_conflict(self) -> None:
        agent = self._agent()
        is_conflict, p_risk = agent._classify(self._cpa(lat=2.0, vert=400.0))
        assert is_conflict is True
        assert p_risk > 0.0

    def test_lateral_only_violated_not_conflict(self) -> None:
        """Lateral violated but vertical >= min: NOT a LoS conflict."""
        agent = self._agent()
        is_conflict, p_risk = agent._classify(self._cpa(lat=2.0, vert=1500.0))
        assert is_conflict is False
        # vert_risk = 0 (1500 >= 1000), so p_risk = min(lat_risk, 0) = 0
        assert p_risk == pytest.approx(0.0)

    def test_vertical_only_violated_not_conflict(self) -> None:
        """Vertical violated but lateral >= min: NOT a LoS conflict."""
        agent = self._agent()
        is_conflict, p_risk = agent._classify(self._cpa(lat=6.0, vert=400.0))
        assert is_conflict is False
        # lat_risk = 0 (6 >= 5), so p_risk = min(0, vert_risk) = 0
        assert p_risk == pytest.approx(0.0)

    def test_exactly_at_min_not_conflict(self) -> None:
        """CPA exactly at the minimum is NOT a violation (strict inequality)."""
        agent = self._agent()
        is_conflict, p_risk = agent._classify(self._cpa(lat=5.0, vert=1000.0))
        assert is_conflict is False
        assert p_risk == pytest.approx(0.0)

    def test_zero_separation_max_risk(self) -> None:
        """Zero lateral and vertical separation → p_risk == 1.0."""
        agent = self._agent()
        is_conflict, p_risk = agent._classify(self._cpa(lat=0.0, vert=0.0))
        assert is_conflict is True
        assert p_risk == pytest.approx(1.0)

    def test_p_risk_bounded(self) -> None:
        """p_risk must always be in [0, 1]."""
        agent = self._agent()
        for lat in [0.0, 2.5, 5.0, 7.0, 10.0]:
            for vert in [0.0, 500.0, 1000.0, 2000.0]:
                _, p_risk = agent._classify(self._cpa(lat=lat, vert=vert))
                assert 0.0 <= p_risk <= 1.0, f"Out of range: lat={lat}, vert={vert}, p_risk={p_risk}"


class TestComputeConflictRisk:
    """Direct unit tests for the shared compute_conflict_risk function."""

    def test_zero_separation_yields_one(self) -> None:
        assert compute_conflict_risk(0.0, 0.0, 5.0, 1000.0) == pytest.approx(1.0)

    def test_at_lateral_minimum_yields_zero(self) -> None:
        assert compute_conflict_risk(5.0, 0.0, 5.0, 1000.0) == pytest.approx(0.0)

    def test_beyond_lateral_minimum_yields_zero(self) -> None:
        assert compute_conflict_risk(7.5, 0.0, 5.0, 1000.0) == pytest.approx(0.0)

    def test_at_vertical_minimum_yields_zero(self) -> None:
        assert compute_conflict_risk(0.0, 1000.0, 5.0, 1000.0) == pytest.approx(0.0)

    def test_linear_interpolation(self) -> None:
        # Halfway on lateral (2.5 NM -> 0.5), quarter on vertical (250 ft -> 0.75) -> min is 0.5
        assert compute_conflict_risk(2.5, 250.0, 5.0, 1000.0) == pytest.approx(0.5)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="lateral_min_nm"):
            compute_conflict_risk(1.0, 100.0, lateral_min_nm=0.0)
        with pytest.raises(ValueError, match="vertical_min_ft"):
            compute_conflict_risk(1.0, 100.0, vertical_min_ft=-10.0)


# ---------------------------------------------------------------------------
# 5. advise_scenario integration tests
# ---------------------------------------------------------------------------


class TestAdviseScenario:
    """End-to-end integration tests using in-memory scenarios."""

    DT = 5.0
    N_STEPS = 200  # 1000 s — enough for history + horizon

    def _agent(self) -> ConflictAdvisoryAgent:
        return ConflictAdvisoryAgent(SmoothedConstantVelocityPredictor())

    def test_converging_scenario_flags_conflict(self) -> None:
        """Two head-on converging aircraft close enough to produce LoS."""
        # AC1 at (0, 0) moving right at 0.05 NM/s → +0.25 NM per step
        # AC2 at (10, 0) moving left at 0.05 NM/s → -0.25 NM per step
        # They will collide in ~100 s (20 steps) → deep LoS well within horizon
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(10.0, 0.0, 20000.0, vx=-0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("conv", {"AC1": track_1, "AC2": track_2})
        agent = self._agent()

        report = agent.advise_scenario(df)

        assert report.has_conflict is True
        assert report.num_flagged > 0
        assert report.max_p_risk > 0.0
        assert report.earliest_lead_time_s is not None
        assert report.earliest_lead_time_s > 0.0

    def test_parallel_tracks_no_conflict(self) -> None:
        """Two parallel aircraft 20 NM apart never violate lateral separation."""
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(20.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("par", {"AC1": track_1, "AC2": track_2})
        agent = self._agent()

        report = agent.advise_scenario(df)

        assert report.has_conflict is False
        assert report.num_flagged == 0
        assert report.earliest_lead_time_s is None

    def test_vertically_separated_no_conflict(self) -> None:
        """Two aircraft laterally close but 5000 ft apart vertically — no LoS."""
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(2.0, 0.0, 25000.0, vx=-0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("vert", {"AC1": track_1, "AC2": track_2})
        agent = self._agent()

        report = agent.advise_scenario(df)

        assert report.has_conflict is False

    def test_report_metadata_is_correct(self) -> None:
        """ScenarioAdvisoryReport metadata fields are populated correctly."""
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(20.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("meta_test", {"AC1": track_1, "AC2": track_2})
        agent = self._agent()

        report = agent.advise_scenario(df)

        assert report.scenario_id == "meta_test"
        assert report.num_aircraft == 2
        assert report.num_origins > 0
        assert len(report.advisories) > 0

    def test_missing_required_column_raises(self) -> None:
        """Missing required column raises ValueError."""
        df = pd.DataFrame({"aircraft_id": ["A", "B"], "t": [0.0, 0.0]})
        agent = self._agent()
        with pytest.raises(ValueError, match="missing required columns"):
            agent.advise_scenario(df)

    def test_single_aircraft_raises(self) -> None:
        """Single aircraft scenario raises ValueError."""
        track = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("single", {"AC1": track})
        agent = self._agent()
        with pytest.raises(ValueError, match="at least 2 aircraft"):
            agent.advise_scenario(df)

    def test_three_aircraft_all_pairs_evaluated(self) -> None:
        """Three aircraft: 3 unique pairs should be evaluated."""
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.02, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(20.0, 0.0, 20000.0, vx=0.02, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_3 = _straight_track(0.0, 20.0, 20000.0, vx=0.02, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("three", {"AC1": track_1, "AC2": track_2, "AC3": track_3})
        agent = self._agent()

        report = agent.advise_scenario(df)

        # Each origin produces 3 pair advisories
        assert report.num_aircraft == 3
        assert len(report.advisories) > 0
        # Number of advisories per origin == number of pairs (C(3,2) = 3)
        assert len(report.advisories) % 3 == 0

    def test_p_risk_max_in_conflict_scenario(self) -> None:
        """max_p_risk > 0 for a conflict scenario."""
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(10.0, 0.0, 20000.0, vx=-0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("risk", {"AC1": track_1, "AC2": track_2})
        agent = self._agent()

        report = agent.advise_scenario(df)
        assert report.max_p_risk > 0.0

    def test_works_with_cv_3step_predictor(self) -> None:
        """Agent works with ConstantVelocityPredictor (cv_3step)."""
        pred = ConstantVelocityPredictor()
        agent = ConflictAdvisoryAgent(pred)
        track_1 = _straight_track(0.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        track_2 = _straight_track(20.0, 0.0, 20000.0, vx=0.05, vy=0.0, vz=0.0, n_steps=self.N_STEPS)
        df = _make_scenario_df("cv3", {"AC1": track_1, "AC2": track_2})
        report = agent.advise_scenario(df)
        assert isinstance(report, ScenarioAdvisoryReport)


# ---------------------------------------------------------------------------
# 6. advise_file test
# ---------------------------------------------------------------------------


class TestAdviseFile:
    """Tests for advise_file — file I/O layer."""

    def _agent(self) -> ConflictAdvisoryAgent:
        return ConflictAdvisoryAgent(SmoothedConstantVelocityPredictor())

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        agent = self._agent()
        with pytest.raises(FileNotFoundError):
            agent.advise_file(tmp_path / "does_not_exist.parquet")

    @pytest.mark.parametrize("data_dir_name", ["synthetic", "synthetic_hard", "synthetic_hard_large"])
    def test_advise_file_sample_parquet(self, data_dir_name: str) -> None:
        """Load a sample parquet from the tracked sample set if available."""
        data_dir = ROOT / "data" / data_dir_name
        parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
        if not parquet_files:
            pytest.skip(f"No sample parquets found in data/{data_dir_name}/")

        agent = self._agent()
        sample_file = parquet_files[0]
        report = agent.advise_file(sample_file)

        assert isinstance(report, ScenarioAdvisoryReport)
        assert report.num_aircraft >= 2
        assert report.num_origins >= 1
        assert report.max_p_risk >= 0.0
