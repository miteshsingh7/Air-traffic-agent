"""Trajectory sliding window generator for trajectory prediction and conflict advisory.

Research simulation only - not for operational use.
Extracts prediction windows:
- Origins every 30 s starting at t=60 s (history_s) and ending at duration - horizon_s
- History: last 60 s (12 steps + current step = 13 steps) of [x_nm, y_nm, alt_ft]
- Future: next 300 s (60 steps) of TRUE [x_nm, y_nm, alt_ft]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowMetadata:
    """Metadata identifying a single aircraft's prediction window sample."""

    scenario_id: str
    aircraft_id: str
    origin_t: float


@dataclass(frozen=True)
class ScenarioOriginGroup:
    """All aircraft tracks in a single scenario at a specific prediction origin."""

    scenario_id: str
    origin_t: float
    aircraft_ids: list[str]
    history: dict[str, np.ndarray]  # acid -> shape (13, 3)
    future_true: dict[str, np.ndarray]  # acid -> shape (60, 3)


def extract_scenario_windows(
    scenario_df: pd.DataFrame,
    origin_step_s: float = 30.0,
    history_s: float = 60.0,
    horizon_s: float = 300.0,
    dt_s: float = 5.0,
    use_noisy_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """Generate prediction origin windows for all aircraft in a scenario.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        Trajectory DataFrame containing 'scenario_id', 'aircraft_id', 't', 'x_nm', 'y_nm', 'alt_ft'.
    origin_step_s : float, default 30.0
        Spacing between prediction origins in seconds.
    history_s : float, default 60.0
        Observation history duration in seconds.
    horizon_s : float, default 300.0
        Prediction horizon duration in seconds.
    dt_s : float, default 5.0
        Resample interval in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[WindowMetadata]]
        history_array: Shape (N, 13, 3)
        future_array: Shape (N, 60, 3)
        metadata: List of WindowMetadata of length N.
    """
    for col in ("aircraft_id", "t", "x_nm", "y_nm", "alt_ft"):
        if col not in scenario_df.columns:
            raise ValueError(f"scenario_df must contain column '{col}'")

    scenario_id = str(scenario_df["scenario_id"].iloc[0]) if "scenario_id" in scenario_df.columns else "unknown"

    t_max = float(scenario_df["t"].max())
    origin_start = history_s
    origin_end = t_max - horizon_s

    if origin_end < origin_start:
        raise ValueError(
            f"Scenario duration ({t_max}s) is too short for history ({history_s}s) + horizon ({horizon_s}s)"
        )

    # Origins every origin_step_s
    origins = np.arange(origin_start, origin_end + 1e-4, origin_step_s)

    history_steps = int(round(history_s / dt_s)) + 1  # 12 + 1 = 13
    future_steps = int(round(horizon_s / dt_s))  # 60

    aircraft_ids = sorted(scenario_df["aircraft_id"].unique())

    history_list: list[np.ndarray] = []
    future_list: list[np.ndarray] = []
    metadata_list: list[WindowMetadata] = []

    # Pre-index aircraft for fast slicing
    aircraft_data: dict[str, pd.DataFrame] = {
        acid: scenario_df[scenario_df["aircraft_id"] == acid].sort_values("t").set_index("t")
        for acid in aircraft_ids
    }

    # Features: history uses noisy observations if present, future strictly uses true tracks
    hist_cols = (
        ["x_obs_nm", "y_obs_nm", "alt_obs_ft"]
        if (use_noisy_history and "x_obs_nm" in scenario_df.columns)
        else ["x_nm", "y_nm", "alt_ft"]
    )
    fut_cols = ["x_nm", "y_nm", "alt_ft"]

    for origin in origins:
        origin_t = float(round(origin, 4))
        # History timestamps: [origin_t - 60, ..., origin_t]
        hist_times = [float(round(origin_t - history_s + i * dt_s, 4)) for i in range(history_steps)]
        # Future timestamps: [origin_t + 5, ..., origin_t + 300]
        fut_times = [float(round(origin_t + (i + 1) * dt_s, 4)) for i in range(future_steps)]

        for acid in aircraft_ids:
            df_ac = aircraft_data[acid]

            # Verify that all timestamps exist in trajectory
            if not (all(t in df_ac.index for t in hist_times) and all(t in df_ac.index for t in fut_times)):
                continue

            hist_arr = df_ac.loc[hist_times, hist_cols].to_numpy(dtype=np.float64)
            fut_arr = df_ac.loc[fut_times, fut_cols].to_numpy(dtype=np.float64)

            history_list.append(hist_arr)
            future_list.append(fut_arr)
            metadata_list.append(
                WindowMetadata(
                    scenario_id=scenario_id,
                    aircraft_id=str(acid),
                    origin_t=origin_t,
                )
            )

    if not history_list:
        return (
            np.empty((0, history_steps, 3), dtype=np.float64),
            np.empty((0, future_steps, 3), dtype=np.float64),
            [],
        )

    history_array = np.stack(history_list, axis=0)
    future_array = np.stack(future_list, axis=0)
    return history_array, future_array, metadata_list


def extract_scenario_origin_groups(
    scenario_df: pd.DataFrame,
    origin_step_s: float = 30.0,
    history_s: float = 60.0,
    horizon_s: float = 300.0,
    dt_s: float = 5.0,
    use_noisy_history: bool = True,
) -> list[ScenarioOriginGroup]:
    """Extract origin groups where all aircraft in the scenario are synchronized.

    Useful for multi-aircraft conflict evaluation at each prediction origin.
    """
    scenario_id = str(scenario_df["scenario_id"].iloc[0]) if "scenario_id" in scenario_df.columns else "unknown"
    t_max = float(scenario_df["t"].max())
    origin_start = history_s
    origin_end = t_max - horizon_s

    if origin_end < origin_start:
        return []

    origins = np.arange(origin_start, origin_end + 1e-4, origin_step_s)
    history_steps = int(round(history_s / dt_s)) + 1
    future_steps = int(round(horizon_s / dt_s))
    aircraft_ids = sorted(scenario_df["aircraft_id"].unique())

    aircraft_data = {
        acid: scenario_df[scenario_df["aircraft_id"] == acid].sort_values("t").set_index("t")
        for acid in aircraft_ids
    }
    hist_cols = (
        ["x_obs_nm", "y_obs_nm", "alt_obs_ft"]
        if (use_noisy_history and "x_obs_nm" in scenario_df.columns)
        else ["x_nm", "y_nm", "alt_ft"]
    )
    fut_cols = ["x_nm", "y_nm", "alt_ft"]
    groups: list[ScenarioOriginGroup] = []

    for origin in origins:
        origin_t = float(round(origin, 4))
        hist_times = [float(round(origin_t - history_s + i * dt_s, 4)) for i in range(history_steps)]
        fut_times = [float(round(origin_t + (i + 1) * dt_s, 4)) for i in range(future_steps)]

        valid_aircraft: list[str] = []
        hist_dict: dict[str, np.ndarray] = {}
        fut_dict: dict[str, np.ndarray] = {}

        for acid in aircraft_ids:
            df_ac = aircraft_data[acid]
            if all(t in df_ac.index for t in hist_times) and all(t in df_ac.index for t in fut_times):
                valid_aircraft.append(acid)
                hist_dict[acid] = df_ac.loc[hist_times, hist_cols].to_numpy(dtype=np.float64)
                fut_dict[acid] = df_ac.loc[fut_times, fut_cols].to_numpy(dtype=np.float64)

        if len(valid_aircraft) >= 2:
            groups.append(
                ScenarioOriginGroup(
                    scenario_id=scenario_id,
                    origin_t=origin_t,
                    aircraft_ids=valid_aircraft,
                    history=hist_dict,
                    future_true=fut_dict,
                )
            )

    return groups


def extract_dataset_windows(
    scenario_files: Sequence[str | Path],
    origin_step_s: float = 30.0,
    history_s: float = 60.0,
    horizon_s: float = 300.0,
    dt_s: float = 5.0,
    use_noisy_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """Extract all prediction windows from multiple scenario parquet files."""
    all_histories: list[np.ndarray] = []
    all_futures: list[np.ndarray] = []
    all_metadata: list[WindowMetadata] = []

    for f in scenario_files:
        df = pd.read_parquet(f)
        h, f_arr, meta = extract_scenario_windows(
            df,
            origin_step_s=origin_step_s,
            history_s=history_s,
            horizon_s=horizon_s,
            dt_s=dt_s,
            use_noisy_history=use_noisy_history,
        )
        if len(meta) > 0:
            all_histories.append(h)
            all_futures.append(f_arr)
            all_metadata.extend(meta)

    if not all_metadata:
        history_steps = int(round(history_s / dt_s)) + 1
        future_steps = int(round(horizon_s / dt_s))
        return (
            np.empty((0, history_steps, 3), dtype=np.float64),
            np.empty((0, future_steps, 3), dtype=np.float64),
            [],
        )

    return (
        np.concatenate(all_histories, axis=0),
        np.concatenate(all_futures, axis=0),
        all_metadata,
    )
