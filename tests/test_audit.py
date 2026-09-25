"""Unit tests for the label audit evaluation module.

Verifies:
- Label audit logic on synthetic scenarios with injection flags
- Accurate discrepancy detection for false positives and false negatives
- Statistical summaries for confirmed conflict distributions
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import TrajectoryConfig
from src.eval.label_audit import AuditReport, run_label_audit


def _create_mock_scenario(
    t_vals: np.ndarray,
    has_conflict: bool,
    scenario_id: str,
) -> pd.DataFrame:
    """Create a 2-aircraft scenario with or without conflict."""
    # AC1 flies along y=0
    ac1 = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "aircraft_id": "AC_0",
            "t": t_vals,
            "x_nm": 0.1 * t_vals,
            "y_nm": np.zeros_like(t_vals),
            "alt_ft": np.full_like(t_vals, 30000.0),
            "heading_deg": np.full_like(t_vals, 90.0),
            "speed_kt": np.full_like(t_vals, 360.0),
        }
    )

    # AC2: if conflict, crosses AC1; if no conflict, separated by 50 NM
    y_offset = 0.0 if has_conflict else 50.0
    x_offset = 10.0 if has_conflict else 0.0
    ac2 = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "aircraft_id": "AC_1",
            "t": t_vals,
            "x_nm": np.full_like(t_vals, x_offset),
            "y_nm": (0.1 * t_vals - 10.0) if has_conflict else np.full_like(t_vals, y_offset),
            "alt_ft": np.full_like(t_vals, 30000.0),
            "heading_deg": np.full_like(t_vals, 0.0),
            "speed_kt": np.full_like(t_vals, 360.0),
        }
    )

    return pd.concat([ac1, ac2], ignore_index=True)


def test_label_audit_perfect_match():
    """Verify audit when injection flags perfectly match geometry results."""
    t_vals = np.arange(0.0, 205.0, 5.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create 2 injected scenarios (with conflict) and 3 clean scenarios (without conflict)
        manifest = {}
        for i in range(2):
            sid = f"scenario_{i:04d}"
            df = _create_mock_scenario(t_vals, has_conflict=True, scenario_id=sid)
            df.to_parquet(tmp_path / f"{sid}.parquet", index=False)
            manifest[sid] = {"injected": True}

        for i in range(2, 5):
            sid = f"scenario_{i:04d}"
            df = _create_mock_scenario(t_vals, has_conflict=False, scenario_id=sid)
            df.to_parquet(tmp_path / f"{sid}.parquet", index=False)
            manifest[sid] = {"injected": False}

        with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        cfg = TrajectoryConfig()
        report: AuditReport = run_label_audit(data_dir=tmp_path, config=cfg)

        assert report.total_scenarios == 5
        assert report.injected_with_conflict == 2
        assert report.injected_without_conflict == 0
        assert report.clean_with_conflict == 0
        assert report.clean_without_conflict == 3
        assert report.total_confirmed_conflict_pairs == 2
        assert report.has_discrepancy is False
        assert report.earliest_conflict_time_dist is not None
        assert report.earliest_conflict_time_dist.min_val == pytest.approx(65.0, abs=1e-3)


def test_label_audit_with_discrepancies():
    """Verify audit detects both unconfirmed injections and accidental conflicts."""
    t_vals = np.arange(0.0, 205.0, 5.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        manifest = {}
        # Scenario 0: injected BUT geometry is clean (failed injection)
        sid0 = "scenario_0000"
        df0 = _create_mock_scenario(t_vals, has_conflict=False, scenario_id=sid0)
        df0.to_parquet(tmp_path / f"{sid0}.parquet", index=False)
        manifest[sid0] = {"injected": True}

        # Scenario 1: clean flag BUT geometry has conflict (accidental conflict)
        sid1 = "scenario_0001"
        df1 = _create_mock_scenario(t_vals, has_conflict=True, scenario_id=sid1)
        df1.to_parquet(tmp_path / f"{sid1}.parquet", index=False)
        manifest[sid1] = {"injected": False}

        with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        cfg = TrajectoryConfig()
        report: AuditReport = run_label_audit(data_dir=tmp_path, config=cfg)

        assert report.total_scenarios == 2
        assert report.injected_with_conflict == 0
        assert report.injected_without_conflict == 1
        assert report.clean_with_conflict == 1
        assert report.clean_without_conflict == 0
        assert report.has_discrepancy is True
