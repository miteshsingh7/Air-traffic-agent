"""Unit tests for sliding prediction window extraction.

Verifies:
- Window shapes: history (13, 3) and future (60, 3)
- Temporal alignment: future immediately follows history (origin_t + 5s)
- Origins: every 30s from t=60s to duration - 300s
- Correct metadata mapping (scenario_id, aircraft_id, origin_t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.windows import (
    extract_scenario_origin_groups,
    extract_scenario_windows,
)


def _build_dummy_scenario(duration_s: float = 1200.0, dt_s: float = 5.0) -> pd.DataFrame:
    """Build simple multi-aircraft scenario DataFrame for testing."""
    times = np.arange(0.0, duration_s + dt_s, dt_s)
    ac1 = pd.DataFrame(
        {
            "scenario_id": "scen_test",
            "aircraft_id": "AC_0",
            "t": times,
            "x_nm": 0.1 * times,
            "y_nm": 0.05 * times,
            "alt_ft": np.full_like(times, 33000.0),
            "heading_deg": np.full_like(times, 60.0),
            "speed_kt": np.full_like(times, 400.0),
        }
    )
    ac2 = pd.DataFrame(
        {
            "scenario_id": "scen_test",
            "aircraft_id": "AC_1",
            "t": times,
            "x_nm": 50.0 + 0.08 * times,
            "y_nm": 20.0 + 0.02 * times,
            "alt_ft": np.full_like(times, 35000.0),
            "heading_deg": np.full_like(times, 75.0),
            "speed_kt": np.full_like(times, 380.0),
        }
    )
    return pd.concat([ac1, ac2], ignore_index=True)


def test_window_shapes_and_origins():
    """Verify history shape (13, 3), future shape (60, 3), and origin interval."""
    df = _build_dummy_scenario(duration_s=1200.0)
    histories, futures, metadata = extract_scenario_windows(
        df,
        origin_step_s=30.0,
        history_s=60.0,
        horizon_s=300.0,
        dt_s=5.0,
    )

    # Number of origins = (1200 - 300 - 60) / 30 + 1 = 840 / 30 + 1 = 29 origins
    # With 2 aircraft, total window samples = 29 * 2 = 58
    assert len(metadata) == 58
    assert histories.shape == (58, 13, 3)
    assert futures.shape == (58, 60, 3)

    # Verify origin values
    origins = sorted(set(m.origin_t for m in metadata))
    assert origins[0] == 60.0
    assert origins[-1] == 900.0
    diffs = [round(o2 - o1, 2) for o1, o2 in zip(origins[:-1], origins[1:])]
    assert all(d == 30.0 for d in diffs)


def test_window_temporal_alignment():
    """Verify future truly follows history without gap or overlap."""
    df = _build_dummy_scenario(duration_s=1200.0)
    histories, futures, metadata = extract_scenario_windows(
        df,
        origin_step_s=30.0,
        history_s=60.0,
        horizon_s=300.0,
        dt_s=5.0,
    )

    for i, meta in enumerate(metadata):
        origin_t = meta.origin_t
        h = histories[i]
        f = futures[i]

        # In _build_dummy_scenario:
        # For AC_0: x(t) = 0.1 * t
        if meta.aircraft_id == "AC_0":
            # History last step is at origin_t
            assert h[-1, 0] == pytest.approx(0.1 * origin_t, abs=1e-5)
            # History first step is at origin_t - 60
            assert h[0, 0] == pytest.approx(0.1 * (origin_t - 60.0), abs=1e-5)
            # Future first step is at origin_t + 5s (immediately follows history)
            assert f[0, 0] == pytest.approx(0.1 * (origin_t + 5.0), abs=1e-5)
            # Future last step is at origin_t + 300s
            assert f[-1, 0] == pytest.approx(0.1 * (origin_t + 300.0), abs=1e-5)


def test_scenario_origin_groups():
    """Verify synchronized origin groups across multiple aircraft."""
    df = _build_dummy_scenario(duration_s=1200.0)
    groups = extract_scenario_origin_groups(df, origin_step_s=30.0)
    assert len(groups) == 29
    first_group = groups[0]
    assert first_group.origin_t == 60.0
    assert set(first_group.aircraft_ids) == {"AC_0", "AC_1"}
    assert first_group.history["AC_0"].shape == (13, 3)
    assert first_group.future_true["AC_0"].shape == (60, 3)
