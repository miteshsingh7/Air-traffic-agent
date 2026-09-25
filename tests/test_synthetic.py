"""Unit tests for synthetic scenario and trajectory generator.

Verifies:
- DataFrame schema and required columns
- Aircraft speed and altitude bounds
- Deliberate conflict injection (~30%)
- Clean scenario generation
- Parquet read/write compatibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.conflict.geometry import compute_scenario_conflicts, has_scenario_conflict
from src.data.synthetic import (
    ScenarioGenerator,
    TrajectoryConfig,
    generate_synthetic_dataset,
)

EXPECTED_COLUMNS = [
    "scenario_id",
    "aircraft_id",
    "t",
    "x_nm",
    "y_nm",
    "alt_ft",
    "heading_deg",
    "speed_kt",
]


def test_trajectory_config_defaults():
    """Verify default values and YAML loading."""
    cfg = TrajectoryConfig()
    assert cfg.lateral_min_nm == 5.0
    assert cfg.vertical_min_ft == 1000.0
    assert cfg.resample_rate_s == 5.0
    assert cfg.history_s == 60.0
    assert cfg.horizon_s == 300.0
    assert cfg.scenario_duration_s == 1200.0
    assert cfg.total_duration_s == 1200.0

    yaml_cfg = TrajectoryConfig.from_yaml("configs/default.yaml")
    assert yaml_cfg.lateral_min_nm == 5.0
    assert yaml_cfg.vertical_min_ft == 1000.0
    assert yaml_cfg.resample_rate_s == 5.0
    assert yaml_cfg.scenario_duration_s == 1200.0


def test_scenario_duration_parameter_and_rendezvous_range():
    """Verify custom scenario duration and rendezvous time range [300, duration - 120]."""
    custom_dur = 800.0
    cfg = TrajectoryConfig(scenario_duration_s=custom_dur, total_duration_s=custom_dur)
    assert cfg.scenario_duration_s == 800.0

    generator = ScenarioGenerator(config=cfg, seed=42)
    scen_df = generator.generate_scenario("test_dur_800", inject_conflict=True)

    # Max time in scenario must equal custom_dur
    assert scen_df["t"].max() == custom_dur

    # Earliest conflict time must be >= 300 - margin and <= (duration - 120) + margin
    conflicts = compute_scenario_conflicts(scen_df)
    assert len(conflicts) >= 1
    for c in conflicts:
        assert c.first_conflict_time >= 250.0  # approach begins slightly before tc
        assert c.first_conflict_time <= (custom_dur - 120.0 + 30.0)


def test_scenario_columns_and_bounds():
    """Verify generated scenario structure, column types, and physical bounds."""
    cfg = TrajectoryConfig()
    generator = ScenarioGenerator(config=cfg, seed=123)
    df = generator.generate_scenario("test_0001", inject_conflict=False)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0

    aircraft_ids = df["aircraft_id"].unique()
    assert 2 <= len(aircraft_ids) <= 8

    # Speeds must be in specified range [200, 500] kt
    assert df["speed_kt"].min() >= 200.0 - 1e-3
    assert df["speed_kt"].max() <= 500.0 + 1e-3

    # Altitudes must be within airspace limits
    assert df["alt_ft"].min() >= cfg.alt_min_ft - 1e-3
    assert df["alt_ft"].max() <= cfg.alt_max_ft + 1e-3

    # Time steps must be spaced by resample_rate_s
    t_vals = sorted(df["t"].unique())
    diffs = [round(t2 - t1, 4) for t1, t2 in zip(t_vals[:-1], t_vals[1:])]
    assert all(d == cfg.resample_rate_s for d in diffs)


def test_conflict_injection_consistency():
    """Verify that inject_conflict=True creates a verified conflict and inject_conflict=False is clean."""
    cfg = TrajectoryConfig()
    generator = ScenarioGenerator(config=cfg, seed=999)

    conflict_scen = generator.generate_scenario("scen_conflict", inject_conflict=True)
    assert has_scenario_conflict(conflict_scen) is True
    conflicts = compute_scenario_conflicts(conflict_scen)
    assert len(conflicts) >= 1
    # Check that the injected pair (AC_0, AC_1) is in conflict
    pair_found = any(
        (c.aircraft_1 == "AC_0" and c.aircraft_2 == "AC_1")
        or (c.aircraft_1 == "AC_1" and c.aircraft_2 == "AC_0")
        for c in conflicts
    )
    assert pair_found is True

    clean_scen = generator.generate_scenario("scen_clean", inject_conflict=False)
    assert has_scenario_conflict(clean_scen) is False
    assert len(compute_scenario_conflicts(clean_scen)) == 0


def test_generate_synthetic_dataset_batch():
    """Verify dataset generation pipeline and parquet output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = generate_synthetic_dataset(
            num_scenarios=10,
            conflict_probability=0.30,
            output_dir=tmp_dir,
            seed=42,
        )

        assert summary["scenario_count"] == 10
        assert summary["parquet_files"] == 10
        assert summary["conflict_scenarios"] == 3
        assert summary["clean_scenarios"] == 7
        assert summary["conflict_fraction"] == pytest.approx(0.30, abs=1e-5)
        assert summary["rows_written"] > 0

        parquet_files = list(Path(tmp_dir).glob("*.parquet"))
        assert len(parquet_files) == 10

        # Read back a parquet file and check schema
        sample_df = pd.read_parquet(parquet_files[0])
        assert list(sample_df.columns) == EXPECTED_COLUMNS
        assert not sample_df.empty
