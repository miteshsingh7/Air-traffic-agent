"""Air-traffic conflict geometry and loss-of-separation (LoS) detection.

Research simulation only - not for operational use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConflictRecord:
    """Record describing a detected conflict between two aircraft in a scenario."""

    scenario_id: str | int
    aircraft_1: str
    aircraft_2: str
    first_conflict_time: float
    min_lateral_distance_nm: float
    min_vertical_distance_ft: float
    conflict_duration_s: float


def _extract_track_columns(track: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """Normalize track input into a DataFrame with standard column names.

    Required logical columns: t, x_nm, y_nm, alt_ft.
    """
    if isinstance(track, pd.DataFrame):
        df = track.copy()
    elif isinstance(track, Mapping):
        df = pd.DataFrame(track)
    else:
        raise TypeError(f"Expected pd.DataFrame or Mapping, got {type(track)}")

    col_map: dict[str, str] = {}
    lower_cols = {col.lower(): col for col in df.columns}

    # Time column resolution
    for candidate in ("t", "time", "timestamp"):
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "t"
            break
    if "t" not in col_map.values():
        raise ValueError(f"Could not identify time column in {list(df.columns)}")

    # X column resolution
    for candidate in ("x_nm", "x", "pos_x"):
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "x_nm"
            break
    if "x_nm" not in col_map.values():
        raise ValueError(f"Could not identify x position column in {list(df.columns)}")

    # Y column resolution
    for candidate in ("y_nm", "y", "pos_y"):
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "y_nm"
            break
    if "y_nm" not in col_map.values():
        raise ValueError(f"Could not identify y position column in {list(df.columns)}")

    # Altitude column resolution
    for candidate in ("alt_ft", "alt", "altitude_ft", "z", "altitude"):
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "alt_ft"
            break
    if "alt_ft" not in col_map.values():
        raise ValueError(f"Could not identify altitude column in {list(df.columns)}")

    renamed = df.rename(columns=col_map)
    return renamed[["t", "x_nm", "y_nm", "alt_ft"]].sort_values("t").reset_index(drop=True)


def find_earliest_conflict(
    track_1: pd.DataFrame | Mapping[str, Any],
    track_2: pd.DataFrame | Mapping[str, Any],
    lateral_min_nm: float = 5.0,
    vertical_min_ft: float = 1000.0,
) -> float | None:
    """Return the earliest time both lateral < 5 NM and vertical < 1000 ft, or None.

    Parameters
    ----------
    track_1 : pd.DataFrame or Mapping[str, Any]
        Trajectory for the first aircraft, containing time, x, y, and altitude.
    track_2 : pd.DataFrame or Mapping[str, Any]
        Trajectory for the second aircraft, containing time, x, y, and altitude.
    lateral_min_nm : float, default 5.0
        Lateral separation minimum in nautical miles.
    vertical_min_ft : float, default 1000.0
        Vertical separation minimum in feet.

    Returns
    -------
    float or None
        The earliest time t where lateral distance < lateral_min_nm and
        vertical distance < vertical_min_ft, or None if no conflict occurs.
    """
    if lateral_min_nm <= 0:
        raise ValueError(f"lateral_min_nm must be positive, got {lateral_min_nm}")
    if vertical_min_ft <= 0:
        raise ValueError(f"vertical_min_ft must be positive, got {vertical_min_ft}")

    df1 = _extract_track_columns(track_1)
    df2 = _extract_track_columns(track_2)

    if df1.empty or df2.empty:
        return None

    # Merge on exact synchronized time steps
    merged = pd.merge(df1, df2, on="t", suffixes=("_1", "_2"))
    if merged.empty:
        return None

    dx = merged["x_nm_1"].to_numpy(dtype=np.float64) - merged["x_nm_2"].to_numpy(dtype=np.float64)
    dy = merged["y_nm_1"].to_numpy(dtype=np.float64) - merged["y_nm_2"].to_numpy(dtype=np.float64)
    lateral_dist = np.hypot(dx, dy)

    alt_1 = merged["alt_ft_1"].to_numpy(dtype=np.float64)
    alt_2 = merged["alt_ft_2"].to_numpy(dtype=np.float64)
    vertical_dist = np.abs(alt_1 - alt_2)

    # Loss of separation strictly requires both lateral < threshold and vertical < threshold
    is_conflict = (lateral_dist < lateral_min_nm) & (vertical_dist < vertical_min_ft)

    if not np.any(is_conflict):
        return None

    conflict_indices = np.where(is_conflict)[0]
    first_idx = conflict_indices[0]
    return float(merged["t"].iloc[first_idx])


def compute_scenario_conflicts(
    scenario_df: pd.DataFrame,
    lateral_min_nm: float = 5.0,
    vertical_min_ft: float = 1000.0,
) -> list[ConflictRecord]:
    """Compute ground-truth conflicts for a full scenario from TRUE trajectories.

    Evaluates all unique unordered pairs of aircraft (AC_i, AC_j) in the scenario.
    For each pair that violates separation standards, computes earliest conflict time,
    minimum lateral distance, minimum vertical distance, and duration of loss of separation.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        DataFrame containing trajectories of all aircraft in the scenario.
        Must contain: 'aircraft_id', 't', 'x_nm', 'y_nm', 'alt_ft'.
        May optionally contain 'scenario_id'.
    lateral_min_nm : float, default 5.0
        Lateral separation minimum in nautical miles.
    vertical_min_ft : float, default 1000.0
        Vertical separation minimum in feet.

    Returns
    -------
    list[ConflictRecord]
        List of ConflictRecord objects describing all pairwise conflicts in the scenario.
    """
    if "aircraft_id" not in scenario_df.columns:
        raise ValueError("scenario_df must contain 'aircraft_id' column")

    scenario_id: str | int = (
        scenario_df["scenario_id"].iloc[0]
        if "scenario_id" in scenario_df.columns and len(scenario_df) > 0
        else "unknown"
    )

    aircraft_ids = sorted(scenario_df["aircraft_id"].unique())
    num_aircraft = len(aircraft_ids)
    conflicts: list[ConflictRecord] = []

    if num_aircraft < 2:
        return conflicts

    # Group tracks by aircraft for efficient lookup
    tracks: dict[str, pd.DataFrame] = {
        acid: scenario_df[scenario_df["aircraft_id"] == acid][["t", "x_nm", "y_nm", "alt_ft"]]
        .sort_values("t")
        .reset_index(drop=True)
        for acid in aircraft_ids
    }

    for i in range(num_aircraft):
        acid_1 = aircraft_ids[i]
        track_1 = tracks[acid_1]
        for j in range(i + 1, num_aircraft):
            acid_2 = aircraft_ids[j]
            track_2 = tracks[acid_2]

            merged = pd.merge(track_1, track_2, on="t", suffixes=("_1", "_2"))
            if merged.empty:
                continue

            dx = merged["x_nm_1"].to_numpy(dtype=np.float64) - merged["x_nm_2"].to_numpy(dtype=np.float64)
            dy = merged["y_nm_1"].to_numpy(dtype=np.float64) - merged["y_nm_2"].to_numpy(dtype=np.float64)
            lateral_dist = np.hypot(dx, dy)

            alt_1 = merged["alt_ft_1"].to_numpy(dtype=np.float64)
            alt_2 = merged["alt_ft_2"].to_numpy(dtype=np.float64)
            vertical_dist = np.abs(alt_1 - alt_2)

            is_conflict = (lateral_dist < lateral_min_nm) & (vertical_dist < vertical_min_ft)

            if not np.any(is_conflict):
                continue

            conflict_indices = np.where(is_conflict)[0]
            first_conflict_time = float(merged["t"].iloc[conflict_indices[0]])

            # Calculate metrics during the conflict window
            conflict_lateral = lateral_dist[conflict_indices]
            conflict_vertical = vertical_dist[conflict_indices]
            min_lat = float(np.min(conflict_lateral))
            min_vert = float(np.min(conflict_vertical))

            # Duration calculation: time between first and last conflict point
            # plus dt if multiple, or step size
            t_conflict = merged["t"].iloc[conflict_indices].to_numpy()
            if len(t_conflict) > 1:
                dt = float(np.median(np.diff(merged["t"].to_numpy())))
                duration_s = float(t_conflict[-1] - t_conflict[0] + dt)
            else:
                duration_s = 5.0

            conflicts.append(
                ConflictRecord(
                    scenario_id=scenario_id,
                    aircraft_1=str(acid_1),
                    aircraft_2=str(acid_2),
                    first_conflict_time=first_conflict_time,
                    min_lateral_distance_nm=min_lat,
                    min_vertical_distance_ft=min_vert,
                    conflict_duration_s=duration_s,
                )
            )

    return conflicts


def has_scenario_conflict(
    scenario_df: pd.DataFrame,
    lateral_min_nm: float = 5.0,
    vertical_min_ft: float = 1000.0,
) -> bool:
    """Return True if any pair of aircraft in the scenario is in conflict."""
    return len(compute_scenario_conflicts(scenario_df, lateral_min_nm, vertical_min_ft)) > 0


def conflicts_to_dataframe(conflicts: Sequence[ConflictRecord]) -> pd.DataFrame:
    """Convert a sequence of ConflictRecord objects to a pandas DataFrame."""
    if not conflicts:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "aircraft_1",
                "aircraft_2",
                "first_conflict_time",
                "min_lateral_distance_nm",
                "min_vertical_distance_ft",
                "conflict_duration_s",
            ]
        )
    return pd.DataFrame([vars(c) for c in conflicts])
