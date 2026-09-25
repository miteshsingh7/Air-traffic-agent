"""Unit tests for conflict geometry and loss-of-separation detection.

Tests the four fundamental cases:
1. Known conflict
2. Known near-miss
3. Known non-conflict
4. Vertical-separated case
along with boundary and multi-aircraft scenario tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.conflict.geometry import (
    ConflictRecord,
    compute_scenario_conflicts,
    conflicts_to_dataframe,
    find_earliest_conflict,
    has_scenario_conflict,
)


def _make_track(
    times: np.ndarray,
    x0: float,
    y0: float,
    vx_kt: float,
    vy_kt: float,
    alt_ft: float,
    vz_fpm: float = 0.0,
    aircraft_id: str = "AC1",
    scenario_id: str = "test_scen",
) -> pd.DataFrame:
    """Helper to build a linear trajectory DataFrame."""
    dt_hr = times / 3600.0
    dt_min = times / 60.0
    x = x0 + vx_kt * dt_hr
    y = y0 + vy_kt * dt_hr
    alt = alt_ft + vz_fpm * dt_min
    speed = np.hypot(vx_kt, vy_kt)
    heading = (np.degrees(np.arctan2(vx_kt, vy_kt)) + 360.0) % 360.0

    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "aircraft_id": aircraft_id,
            "t": times,
            "x_nm": x,
            "y_nm": y,
            "alt_ft": alt,
            "heading_deg": np.full_like(times, heading),
            "speed_kt": np.full_like(times, speed),
        }
    )


def test_known_conflict():
    """Two aircraft on crossing paths at same altitude with minimum distance < 5 NM.

    AC1: flies east along y = 0 from x = 0 at 360 kt (0.1 NM/s).
    AC2: flies north along x = 10 from y = -10 at 360 kt (0.1 NM/s).
    At t = 100 s, both are at (10, 0) at altitude 30,000 ft.
    Distance d(t) = sqrt(2) * |0.1*t - 10|.
    d(t) < 5.0 NM <=> |t - 100| < 35.355 s.
    For 5 s timesteps:
    t = 60 s -> d = sqrt(2) * 4.0 = 5.657 NM (no conflict)
    t = 65 s -> d = sqrt(2) * 3.5 = 4.950 NM (< 5.0 NM, conflict begins!)
    """
    times = np.arange(0, 205, 5, dtype=float)
    # AC1: x0=0, y0=0, vx=360 kt, vy=0 kt
    track1 = _make_track(times, x0=0.0, y0=0.0, vx_kt=360.0, vy_kt=0.0, alt_ft=30000.0, aircraft_id="AC1")
    # AC2: x0=10, y0=-10, vx=0 kt, vy=360 kt
    track2 = _make_track(times, x0=10.0, y0=-10.0, vx_kt=0.0, vy_kt=360.0, alt_ft=30000.0, aircraft_id="AC2")

    earliest_t = find_earliest_conflict(track1, track2)
    assert earliest_t is not None, "Expected a detected conflict"
    assert earliest_t == pytest.approx(65.0, abs=1e-5), f"Expected earliest conflict at t=65s, got {earliest_t}"


def test_known_near_miss():
    """Two aircraft flying at same altitude, closest approach is strictly >= 5.0 NM.

    AC1: flies east along y = 0 at 360 kt.
    AC2: flies east along y = 5.5 NM at 360 kt.
    Lateral distance is constant 5.5 NM >= 5.0 NM.
    Even though vertical separation is 0 ft, lateral separation is never violated.
    Must return None.
    """
    times = np.arange(0, 205, 5, dtype=float)
    track1 = _make_track(times, x0=0.0, y0=0.0, vx_kt=360.0, vy_kt=0.0, alt_ft=33000.0, aircraft_id="AC1")
    track2 = _make_track(times, x0=0.0, y0=5.5, vx_kt=360.0, vy_kt=0.0, alt_ft=33000.0, aircraft_id="AC2")

    earliest_t = find_earliest_conflict(track1, track2)
    assert earliest_t is None, f"Expected no conflict for near-miss (5.5 NM lateral), got {earliest_t}"


def test_known_non_conflict():
    """Two aircraft separated by a large distance (30 NM), flying parallel.

    AC1 at y = 0, AC2 at y = 30 NM.
    Lateral distance is always 30 NM >> 5 NM.
    Must return None.
    """
    times = np.arange(0, 205, 5, dtype=float)
    track1 = _make_track(times, x0=10.0, y0=10.0, vx_kt=400.0, vy_kt=0.0, alt_ft=35000.0, aircraft_id="AC1")
    track2 = _make_track(times, x0=10.0, y0=40.0, vx_kt=400.0, vy_kt=0.0, alt_ft=35000.0, aircraft_id="AC2")

    earliest_t = find_earliest_conflict(track1, track2)
    assert earliest_t is None, f"Expected no conflict for well-separated flights, got {earliest_t}"


def test_vertical_separated_case():
    """Two aircraft cross at the exact same horizontal position, but separated by 2000 ft.

    AC1: at altitude 30,000 ft.
    AC2: at altitude 32,000 ft.
    Vertical difference is 2,000 ft >= 1,000 ft minimum separation.
    Even when lateral distance is 0 NM, loss-of-separation is NOT violated.
    Must return None.
    """
    times = np.arange(0, 205, 5, dtype=float)
    # Both paths cross at (10, 0) at t=100s
    track1 = _make_track(times, x0=0.0, y0=0.0, vx_kt=360.0, vy_kt=0.0, alt_ft=30000.0, aircraft_id="AC1")
    track2 = _make_track(times, x0=10.0, y0=-10.0, vx_kt=0.0, vy_kt=360.0, alt_ft=32000.0, aircraft_id="AC2")

    earliest_t = find_earliest_conflict(track1, track2)
    assert earliest_t is None, (
        f"Expected no conflict due to 2000 ft vertical separation, got {earliest_t}"
    )


def test_boundary_conditions():
    """Test exact boundary conditions where separation is equal to minima."""
    # Boundary: lateral exactly 5.0 NM, alt exactly equal -> Not a conflict (< 5.0 required)
    df_a = pd.DataFrame({"t": [0.0], "x_nm": [0.0], "y_nm": [0.0], "alt_ft": [30000.0]})
    df_b = pd.DataFrame({"t": [0.0], "x_nm": [5.0], "y_nm": [0.0], "alt_ft": [30000.0]})
    assert find_earliest_conflict(df_a, df_b) is None

    # Boundary: lateral 0.0 NM, alt difference exactly 1000.0 ft -> Not a conflict (< 1000 required)
    df_c = pd.DataFrame({"t": [0.0], "x_nm": [0.0], "y_nm": [0.0], "alt_ft": [31000.0]})
    assert find_earliest_conflict(df_a, df_c) is None

    # Strictly inside: lateral 4.99 NM, vertical 999 ft -> Conflict!
    df_d = pd.DataFrame({"t": [0.0], "x_nm": [4.99], "y_nm": [0.0], "alt_ft": [30999.0]})
    assert find_earliest_conflict(df_a, df_d) == 0.0


def test_compute_scenario_conflicts():
    """Test full scenario evaluation with 3 aircraft (AC1 and AC2 conflict, AC3 separated)."""
    times = np.arange(0, 205, 5, dtype=float)
    # AC1 and AC2 conflict at t=65s
    ac1 = _make_track(times, x0=0.0, y0=0.0, vx_kt=360.0, vy_kt=0.0, alt_ft=30000.0, aircraft_id="AC1", scenario_id="scen_01")
    ac2 = _make_track(times, x0=10.0, y0=-10.0, vx_kt=0.0, vy_kt=360.0, alt_ft=30000.0, aircraft_id="AC2", scenario_id="scen_01")
    # AC3 is safely cruising at (100, 100) at 39000 ft
    ac3 = _make_track(times, x0=100.0, y0=100.0, vx_kt=450.0, vy_kt=0.0, alt_ft=39000.0, aircraft_id="AC3", scenario_id="scen_01")

    scenario_df = pd.concat([ac1, ac2, ac3], ignore_index=True)

    conflicts = compute_scenario_conflicts(scenario_df)
    assert len(conflicts) == 1, f"Expected exactly 1 conflict pair, got {len(conflicts)}"

    conf = conflicts[0]
    assert conf.aircraft_1 == "AC1"
    assert conf.aircraft_2 == "AC2"
    assert conf.first_conflict_time == pytest.approx(65.0, abs=1e-5)
    assert conf.min_lateral_distance_nm < 5.0
    assert conf.min_vertical_distance_ft < 1000.0
    assert conf.conflict_duration_s > 0

    assert has_scenario_conflict(scenario_df) is True

    # Test conversion to dataframe
    df_conf = conflicts_to_dataframe(conflicts)
    assert len(df_conf) == 1
    assert "aircraft_1" in df_conf.columns
    assert "first_conflict_time" in df_conf.columns


def test_empty_or_mismatched_tracks():
    """Verify robust handling of empty DataFrames or non-overlapping time intervals."""
    df_empty = pd.DataFrame(columns=["t", "x_nm", "y_nm", "alt_ft"])
    df_valid = pd.DataFrame({"t": [0, 5], "x_nm": [0, 1], "y_nm": [0, 1], "alt_ft": [30000, 30000]})
    assert find_earliest_conflict(df_empty, df_valid) is None

    # Non-overlapping times
    df_t1 = pd.DataFrame({"t": [0, 5], "x_nm": [0, 1], "y_nm": [0, 1], "alt_ft": [30000, 30000]})
    df_t2 = pd.DataFrame({"t": [10, 15], "x_nm": [0, 1], "y_nm": [0, 1], "alt_ft": [30000, 30000]})
    assert find_earliest_conflict(df_t1, df_t2) is None
