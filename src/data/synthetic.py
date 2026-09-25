"""Synthetic trajectory and scenario generator with conflict injection.

Research simulation only - not for operational use.
Generates multi-aircraft local airspace scenarios with realistic kinematics,
gentle turns, climb/descent, and controlled conflict injection (~30%).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.conflict.geometry import compute_scenario_conflicts, has_scenario_conflict


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration parameters for synthetic trajectory simulation."""

    lateral_min_nm: float = 5.0
    vertical_min_ft: float = 1000.0
    resample_rate_s: float = 5.0
    history_s: float = 60.0
    horizon_s: float = 300.0
    scenario_duration_s: float = 1200.0
    total_duration_s: float = 1200.0
    x_min_nm: float = 0.0
    x_max_nm: float = 200.0
    y_min_nm: float = 0.0
    y_max_nm: float = 200.0
    alt_min_ft: float = 10000.0
    alt_max_ft: float = 40000.0
    min_aircraft: int = 2
    max_aircraft: int = 8
    min_speed_kt: float = 200.0
    max_speed_kt: float = 500.0
    conflict_probability: float = 0.30
    default_num_scenarios: int = 500
    output_dir: str = "data/synthetic"
    dataset_variant: str = "easy"
    maneuver_conflict_prob: float = 0.50
    near_miss_scenario_prob: float = 0.25
    noise_sigma_horizontal_nm: float = 0.0
    noise_sigma_vertical_ft: float = 0.0
    splits_file: str = "data/splits.json"
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, variant: str | None = None) -> TrajectoryConfig:
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        target_variant = variant or data.get("dataset_variant", "easy")
        sep = data.get("separation_minima", {})
        time_cfg = data.get("time", {})
        airspace = data.get("airspace", {})
        sim = data.get("simulation", {})

        dur = float(time_cfg.get("scenario_duration_s", time_cfg.get("total_duration_s", 1200.0)))

        if target_variant == "hard_large":
            hard_cfg = data.get("hard_large_dataset", {})
            out_dir = str(hard_cfg.get("output_dir", "data/synthetic_hard_large"))
            splits_f = str(hard_cfg.get("splits_file", "data/splits_hard_large.json"))
            maneuver_p = float(hard_cfg.get("maneuver_conflict_prob", 0.50))
            near_miss_p = float(hard_cfg.get("near_miss_scenario_prob", 0.25))
            noise_h = float(hard_cfg.get("history_noise_sigma_horizontal_nm", 0.02))
            noise_v = float(hard_cfg.get("history_noise_sigma_vertical_ft", 50.0))
            seed_val = int(hard_cfg.get("seed", 2042))
            def_scenarios = int(hard_cfg.get("default_num_scenarios", 3000))
            sim_variant = "hard"
        elif target_variant == "hard":
            hard_cfg = data.get("hard_dataset", {})
            out_dir = str(hard_cfg.get("output_dir", "data/synthetic_hard"))
            splits_f = str(hard_cfg.get("splits_file", "data/splits_hard.json"))
            maneuver_p = float(hard_cfg.get("maneuver_conflict_prob", 0.50))
            near_miss_p = float(hard_cfg.get("near_miss_scenario_prob", 0.25))
            noise_h = float(hard_cfg.get("history_noise_sigma_horizontal_nm", 0.02))
            noise_v = float(hard_cfg.get("history_noise_sigma_vertical_ft", 50.0))
            seed_val = int(hard_cfg.get("seed", 1042))
            def_scenarios = int(sim.get("default_num_scenarios", 500))
            sim_variant = "hard"
        else:
            out_dir = str(sim.get("output_dir", "data/synthetic"))
            splits_f = str(data.get("splits", {}).get("output_file", "data/splits.json"))
            maneuver_p = 0.0
            near_miss_p = 0.0
            noise_h = 0.0
            noise_v = 0.0
            seed_val = int(data.get("splits", {}).get("seed", 42))
            def_scenarios = int(sim.get("default_num_scenarios", 500))
            sim_variant = "easy"

        return cls(
            lateral_min_nm=float(sep.get("lateral_nm", 5.0)),
            vertical_min_ft=float(sep.get("vertical_ft", 1000.0)),
            resample_rate_s=float(time_cfg.get("resample_rate_s", 5.0)),
            history_s=float(time_cfg.get("history_s", 60.0)),
            horizon_s=float(time_cfg.get("horizon_s", 300.0)),
            scenario_duration_s=dur,
            total_duration_s=dur,
            x_min_nm=float(airspace.get("x_min_nm", 0.0)),
            x_max_nm=float(airspace.get("x_max_nm", 200.0)),
            y_min_nm=float(airspace.get("y_min_nm", 0.0)),
            y_max_nm=float(airspace.get("y_max_nm", 200.0)),
            alt_min_ft=float(airspace.get("alt_min_ft", 10000.0)),
            alt_max_ft=float(airspace.get("alt_max_ft", 40000.0)),
            min_aircraft=int(sim.get("min_aircraft", 2)),
            max_aircraft=int(sim.get("max_aircraft", 8)),
            min_speed_kt=float(sim.get("min_speed_kt", 200.0)),
            max_speed_kt=float(sim.get("max_speed_kt", 500.0)),
            conflict_probability=float(sim.get("conflict_probability", 0.30)),
            default_num_scenarios=def_scenarios,
            output_dir=out_dir,
            dataset_variant=sim_variant,
            maneuver_conflict_prob=maneuver_p,
            near_miss_scenario_prob=near_miss_p,
            noise_sigma_horizontal_nm=noise_h,
            noise_sigma_vertical_ft=noise_v,
            splits_file=splits_f,
            seed=seed_val,
        )


class ScenarioGenerator:
    """Generator for multi-aircraft synthetic scenarios with conflict injection."""

    def __init__(self, config: TrajectoryConfig | None = None, seed: int = 42) -> None:
        self.config = config or TrajectoryConfig()
        self.rng = np.random.default_rng(seed)

    def _simulate_single_track(
        self,
        t_steps: np.ndarray,
        x0: float,
        y0: float,
        alt0: float,
        heading0_deg: float,
        speed_kt: float,
        turn_rate_dps: float = 0.0,
        turn_start_s: float = 0.0,
        turn_duration_s: float = 0.0,
        vz_fpm: float = 0.0,
        vz_start_s: float = 0.0,
        vz_duration_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate kinematic trajectory over specified time steps."""
        n = len(t_steps)
        dt = float(self.config.resample_rate_s)

        x = np.zeros(n, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        alt = np.zeros(n, dtype=np.float64)
        heading = np.zeros(n, dtype=np.float64)
        speed = np.full(n, speed_kt, dtype=np.float64)

        cur_x = x0
        cur_y = y0
        cur_alt = alt0
        cur_heading = heading0_deg

        vz_fps = vz_fpm / 60.0

        for i, t in enumerate(t_steps):
            x[i] = cur_x
            y[i] = cur_y
            alt[i] = cur_alt
            heading[i] = cur_heading % 360.0

            # Update heading if inside turn maneuver window
            is_turning = turn_start_s <= t < (turn_start_s + turn_duration_s)
            d_heading = (turn_rate_dps * dt) if is_turning else 0.0
            cur_heading = (cur_heading + d_heading) % 360.0

            # Update altitude if inside vertical maneuver window
            is_climbing = vz_start_s <= t < (vz_start_s + vz_duration_s)
            d_alt = (vz_fps * dt) if is_climbing else 0.0
            cur_alt = float(
                np.clip(
                    cur_alt + d_alt,
                    self.config.alt_min_ft,
                    self.config.alt_max_ft,
                )
            )

            # Advance horizontal position
            # Aviation heading: 0=N (+y), 90=E (+x), 180=S (-y), 270=W (-x)
            heading_rad = np.radians(cur_heading)
            v_nm_s = speed_kt / 3600.0
            cur_x += v_nm_s * np.sin(heading_rad) * dt
            cur_y += v_nm_s * np.cos(heading_rad) * dt

        return x, y, alt, heading, speed

    def generate_scenario(
        self,
        scenario_id: str,
        inject_conflict: bool = False,
        return_metadata: bool = False,
        is_maneuvering: bool | None = None,
        maneuver_type: str | None = None,
        is_near_miss: bool | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """Generate a single scenario with 2-8 aircraft and optional conflict injection."""
        cfg = self.config
        t_steps = np.arange(0.0, cfg.scenario_duration_s + cfg.resample_rate_s, cfg.resample_rate_s)
        dt = cfg.resample_rate_s
        num_aircraft = int(self.rng.integers(cfg.min_aircraft, cfg.max_aircraft + 1))

        # Determine scenario parameters
        if inject_conflict:
            if is_maneuvering is None:
                should_maneuver = (
                    cfg.dataset_variant == "hard"
                    and self.rng.random() < cfg.maneuver_conflict_prob
                )
            else:
                should_maneuver = bool(is_maneuvering)

            if should_maneuver:
                if maneuver_type is None:
                    # ~1/3 climb_level_off, ~2/3 turn
                    chosen_mtype = "climb_level_off" if self.rng.random() < (1.0 / 3.0) else "turn"
                else:
                    chosen_mtype = maneuver_type
            else:
                chosen_mtype = None
            should_near_miss = False
        else:
            should_maneuver = False
            chosen_mtype = None
            if is_near_miss is None:
                should_near_miss = (
                    cfg.dataset_variant == "hard"
                    and self.rng.random() < cfg.near_miss_scenario_prob
                )
            else:
                should_near_miss = bool(is_near_miss)

        # Max retries to ensure ground-truth consistency
        for _ in range(40):
            aircraft_records: list[dict[str, Any]] = []

            if inject_conflict:
                # 1. Target conflict meeting parameters
                t_conflict = float(self.rng.uniform(300.0, cfg.scenario_duration_s - 120.0))
                meeting_x = float(self.rng.uniform(cfg.x_min_nm + 60.0, cfg.x_max_nm - 60.0))
                meeting_y = float(self.rng.uniform(cfg.y_min_nm + 60.0, cfg.y_max_nm - 60.0))
                meeting_alt = float(self.rng.uniform(22000.0, 36000.0))

                v0 = float(self.rng.uniform(320.0, 480.0))
                t_idx_c = int(round(t_conflict / dt))

                vz0_rate = 0.0
                vz0_start = 0.0
                vz0_dur = 0.0

                if should_maneuver and chosen_mtype == "turn":
                    # Maneuvering conflict (turn): heading change before conflict
                    turn_angle = float(self.rng.uniform(20.0, 60.0)) * float(self.rng.choice([-1.0, 1.0]))
                    desired_rate = float(self.rng.uniform(1.5, 3.0))
                    n_turn_steps = max(3, int(round(abs(turn_angle) / (desired_rate * dt))))
                    turn_dur = float(n_turn_steps * dt)
                    turn_rate = float(turn_angle / turn_dur)

                    latest_start = max(90.0, t_conflict - 60.0 - turn_dur)
                    if latest_start <= 90.0:
                        t_maneuver = 90.0
                    else:
                        t_raw = float(self.rng.uniform(90.0, latest_start))
                        t_maneuver = float(round(t_raw / dt) * dt)

                    final_heading = float(self.rng.uniform(0.0, 360.0))
                    init_heading = (final_heading - turn_angle) % 360.0

                    trial_x, trial_y, _, _, _ = self._simulate_single_track(
                        t_steps=t_steps,
                        x0=0.0,
                        y0=0.0,
                        alt0=meeting_alt,
                        heading0_deg=init_heading,
                        speed_kt=v0,
                        turn_rate_dps=turn_rate,
                        turn_start_s=t_maneuver,
                        turn_duration_s=turn_dur,
                    )
                    x0 = meeting_x - trial_x[t_idx_c]
                    y0 = meeting_y - trial_y[t_idx_c]
                    alt0 = meeting_alt
                    heading0 = init_heading
                    turn0_rate = turn_rate
                    turn0_start = t_maneuver
                    turn0_dur = turn_dur
                elif should_maneuver and chosen_mtype == "climb_level_off":
                    # Maneuvering conflict (climb/level-off):
                    # AC_0 climbs or descends and levels off at meeting altitude before rendezvous
                    heading0 = float(self.rng.uniform(0.0, 360.0))
                    d0_to_meeting = (v0 / 3600.0) * t_conflict
                    x0 = meeting_x - d0_to_meeting * np.sin(np.radians(heading0))
                    y0 = meeting_y - d0_to_meeting * np.cos(np.radians(heading0))
                    turn0_rate = 0.0
                    turn0_start = 0.0
                    turn0_dur = 0.0

                    vz_dur = float(self.rng.uniform(45.0, 90.0))
                    vz_steps = max(3, int(round(vz_dur / dt)))
                    vz_dur = float(vz_steps * dt)
                    vz_sign = float(self.rng.choice([-1.0, 1.0]))
                    vz_rate_fpm = float(self.rng.uniform(1200.0, 2400.0)) * vz_sign
                    delta_alt = (vz_rate_fpm / 60.0) * vz_dur
                    alt0 = meeting_alt - delta_alt

                    if alt0 < cfg.alt_min_ft + 1000.0 or alt0 > cfg.alt_max_ft - 1000.0:
                        vz_rate_fpm = -vz_rate_fpm
                        delta_alt = (vz_rate_fpm / 60.0) * vz_dur
                        alt0 = meeting_alt - delta_alt
                    if alt0 < cfg.alt_min_ft + 1000.0 or alt0 > cfg.alt_max_ft - 1000.0:
                        alt0 = float(np.clip(alt0, cfg.alt_min_ft + 1000.0, cfg.alt_max_ft - 1000.0))
                        delta_alt = meeting_alt - alt0
                        vz_rate_fpm = (delta_alt / vz_dur) * 60.0

                    latest_vz_start = max(90.0, t_conflict - 60.0 - vz_dur)
                    if latest_vz_start <= 90.0:
                        t_vz_start = 90.0
                    else:
                        t_vz_start = float(round(self.rng.uniform(90.0, latest_vz_start) / dt) * dt)

                    vz0_rate = vz_rate_fpm
                    vz0_start = t_vz_start
                    vz0_dur = vz_dur
                else:
                    heading0 = float(self.rng.uniform(0.0, 360.0))
                    d0_to_meeting = (v0 / 3600.0) * t_conflict
                    x0 = meeting_x - d0_to_meeting * np.sin(np.radians(heading0))
                    y0 = meeting_y - d0_to_meeting * np.cos(np.radians(heading0))
                    alt0 = meeting_alt
                    turn0_rate = float(self.rng.uniform(-1.5, 1.5))
                    turn0_start = t_conflict + float(self.rng.uniform(15.0, 30.0))
                    turn0_dur = float(self.rng.uniform(15.0, 30.0))

                x_arr, y_arr, alt_arr, h_arr, spd_arr = self._simulate_single_track(
                    t_steps=t_steps,
                    x0=x0,
                    y0=y0,
                    alt0=alt0,
                    heading0_deg=heading0,
                    speed_kt=v0,
                    turn_rate_dps=turn0_rate,
                    turn_start_s=turn0_start,
                    turn_duration_s=turn0_dur,
                    vz_fpm=vz0_rate,
                    vz_start_s=vz0_start,
                    vz_duration_s=vz0_dur,
                )

                for t, x_nm, y_nm, alt_ft, h_deg, s_kt in zip(
                    t_steps, x_arr, y_arr, alt_arr, h_arr, spd_arr
                ):
                    aircraft_records.append(
                        {
                            "scenario_id": scenario_id,
                            "aircraft_id": "AC_0",
                            "t": float(t),
                            "x_nm": round(float(x_nm), 4),
                            "y_nm": round(float(y_nm), 4),
                            "alt_ft": round(float(alt_ft), 1),
                            "heading_deg": round(float(h_deg), 2),
                            "speed_kt": round(float(s_kt), 2),
                        }
                    )

                # AC_1: inbound to meeting point with small miss distance (< 5 NM lateral, < 1000 ft vertical)
                v1 = float(self.rng.uniform(320.0, 480.0))
                crossing_angle = float(self.rng.uniform(60.0, 140.0)) * float(self.rng.choice([-1.0, 1.0]))
                effective_h0 = final_heading if (should_maneuver and chosen_mtype == "turn") else heading0
                heading1 = (effective_h0 + crossing_angle) % 360.0

                miss_lat = float(self.rng.uniform(0.5, 3.2))  # Strictly < 5.0 NM
                miss_angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
                dx_miss = miss_lat * np.sin(miss_angle)
                dy_miss = miss_lat * np.cos(miss_angle)
                dz_miss = float(self.rng.uniform(-400.0, 400.0))  # Strictly < 1000 ft

                target1_x = meeting_x + dx_miss
                target1_y = meeting_y + dy_miss
                target1_alt = meeting_alt + dz_miss

                d1_to_meeting = (v1 / 3600.0) * t_conflict
                x1 = target1_x - d1_to_meeting * np.sin(np.radians(heading1))
                y1 = target1_y - d1_to_meeting * np.cos(np.radians(heading1))
                alt1 = target1_alt

                turn1_rate = float(self.rng.uniform(-1.5, 1.5))
                turn1_start = t_conflict + float(self.rng.uniform(15.0, 30.0))
                turn1_dur = float(self.rng.uniform(15.0, 30.0))

                x_arr, y_arr, alt_arr, h_arr, spd_arr = self._simulate_single_track(
                    t_steps=t_steps,
                    x0=x1,
                    y0=y1,
                    alt0=alt1,
                    heading0_deg=heading1,
                    speed_kt=v1,
                    turn_rate_dps=turn1_rate,
                    turn_start_s=turn1_start,
                    turn_duration_s=turn1_dur,
                )

                for t, x_nm, y_nm, alt_ft, h_deg, s_kt in zip(
                    t_steps, x_arr, y_arr, alt_arr, h_arr, spd_arr
                ):
                    aircraft_records.append(
                        {
                            "scenario_id": scenario_id,
                            "aircraft_id": "AC_1",
                            "t": float(t),
                            "x_nm": round(float(x_nm), 4),
                            "y_nm": round(float(y_nm), 4),
                            "alt_ft": round(float(alt_ft), 1),
                            "heading_deg": round(float(h_deg), 2),
                            "speed_kt": round(float(s_kt), 2),
                        }
                    )

                start_idx = 2
                closest_lat = None
                closest_vert = None
            else:
                # Clean scenario: optionally inject near-miss negative encounter
                if should_near_miss:
                    t_near = float(self.rng.uniform(300.0, cfg.scenario_duration_s - 120.0))
                    near_x = float(self.rng.uniform(cfg.x_min_nm + 60.0, cfg.x_max_nm - 60.0))
                    near_y = float(self.rng.uniform(cfg.y_min_nm + 60.0, cfg.y_max_nm - 60.0))
                    near_alt = float(self.rng.uniform(22000.0, 36000.0))

                    v0 = float(self.rng.uniform(320.0, 480.0))
                    heading0 = float(self.rng.uniform(0.0, 360.0))
                    d0 = (v0 / 3600.0) * t_near
                    x0 = near_x - d0 * np.sin(np.radians(heading0))
                    y0 = near_y - d0 * np.cos(np.radians(heading0))

                    x_arr, y_arr, alt_arr, h_arr, spd_arr = self._simulate_single_track(
                        t_steps=t_steps,
                        x0=x0,
                        y0=y0,
                        alt0=near_alt,
                        heading0_deg=heading0,
                        speed_kt=v0,
                    )
                    for t, x_nm, y_nm, alt_ft, h_deg, s_kt in zip(
                        t_steps, x_arr, y_arr, alt_arr, h_arr, spd_arr
                    ):
                        aircraft_records.append(
                            {
                                "scenario_id": scenario_id,
                                "aircraft_id": "AC_0",
                                "t": float(t),
                                "x_nm": round(float(x_nm), 4),
                                "y_nm": round(float(y_nm), 4),
                                "alt_ft": round(float(alt_ft), 1),
                                "heading_deg": round(float(h_deg), 2),
                                "speed_kt": round(float(s_kt), 2),
                            }
                        )

                    # AC_1: near-miss offset (5.6-8.5 NM lateral OR 1150-1800 ft vertical)
                    v1 = float(self.rng.uniform(320.0, 480.0))
                    cross_ang = float(self.rng.uniform(60.0, 140.0)) * float(self.rng.choice([-1.0, 1.0]))
                    heading1 = (heading0 + cross_ang) % 360.0

                    if self.rng.random() < 0.5:
                        # Lateral near-miss: 5.6 to 8.5 NM (strictly >= 5.0 NM, target >= 5.5 NM)
                        miss_lat = float(self.rng.uniform(5.6, 8.5))
                        miss_ang = float(self.rng.uniform(0.0, 2.0 * np.pi))
                        dx_near = miss_lat * np.sin(miss_ang)
                        dy_near = miss_lat * np.cos(miss_ang)
                        dz_near = float(self.rng.uniform(-400.0, 400.0))
                    else:
                        # Vertical near-miss: 1150 to 1800 ft (strictly >= 1000 ft, target >= 1100 ft)
                        miss_lat = float(self.rng.uniform(0.5, 3.5))
                        miss_ang = float(self.rng.uniform(0.0, 2.0 * np.pi))
                        dx_near = miss_lat * np.sin(miss_ang)
                        dy_near = miss_lat * np.cos(miss_ang)
                        dz_near = float(self.rng.uniform(1150.0, 1800.0)) * float(self.rng.choice([-1.0, 1.0]))

                    target_near1_x = near_x + dx_near
                    target_near1_y = near_y + dy_near
                    target_near1_alt = near_alt + dz_near

                    d1 = (v1 / 3600.0) * t_near
                    x1 = target_near1_x - d1 * np.sin(np.radians(heading1))
                    y1 = target_near1_y - d1 * np.cos(np.radians(heading1))

                    x_arr, y_arr, alt_arr, h_arr, spd_arr = self._simulate_single_track(
                        t_steps=t_steps,
                        x0=x1,
                        y0=y1,
                        alt0=target_near1_alt,
                        heading0_deg=heading1,
                        speed_kt=v1,
                    )
                    for t, x_nm, y_nm, alt_ft, h_deg, s_kt in zip(
                        t_steps, x_arr, y_arr, alt_arr, h_arr, spd_arr
                    ):
                        aircraft_records.append(
                            {
                                "scenario_id": scenario_id,
                                "aircraft_id": "AC_1",
                                "t": float(t),
                                "x_nm": round(float(x_nm), 4),
                                "y_nm": round(float(y_nm), 4),
                                "alt_ft": round(float(alt_ft), 1),
                                "heading_deg": round(float(h_deg), 2),
                                "speed_kt": round(float(s_kt), 2),
                            }
                        )

                    start_idx = 2
                else:
                    start_idx = 0

            # Generate remaining aircraft (start_idx to num_aircraft)
            # Safe altitude tiers and dispersed positions
            used_alts = [
                meeting_alt if inject_conflict else -99999.0,
            ]
            for ac_num in range(start_idx, num_aircraft):
                acid = f"AC_{ac_num}"
                v_kt = float(self.rng.uniform(cfg.min_speed_kt, cfg.max_speed_kt))
                h_deg = float(self.rng.uniform(0.0, 360.0))
                x_init = float(self.rng.uniform(cfg.x_min_nm + 25.0, cfg.x_max_nm - 25.0))
                y_init = float(self.rng.uniform(cfg.y_min_nm + 25.0, cfg.y_max_nm - 25.0))

                # Pick altitude separated by at least 2000 ft from used altitudes
                # or standard flight levels
                candidate_alts = np.arange(14000.0, 39000.0, 2000.0)
                safe_alts = [
                    a for a in candidate_alts if all(abs(a - ua) >= 2000.0 for ua in used_alts)
                ]
                if safe_alts:
                    alt_init = float(self.rng.choice(safe_alts))
                else:
                    alt_init = float(self.rng.uniform(cfg.alt_min_ft + 1000.0, cfg.alt_max_ft - 1000.0))
                used_alts.append(alt_init)

                # Optional gentle turn (30% probability)
                has_turn = self.rng.random() < 0.35
                turn_rate = float(self.rng.uniform(-1.5, 1.5)) if has_turn else 0.0
                turn_start = float(self.rng.uniform(30.0, cfg.scenario_duration_s - 60.0))
                turn_dur = float(self.rng.uniform(15.0, 45.0))

                # Optional gentle climb/descent (25% probability)
                has_climb = self.rng.random() < 0.25
                vz = float(self.rng.uniform(-1200.0, 1200.0)) if has_climb else 0.0
                vz_start = float(self.rng.uniform(30.0, cfg.scenario_duration_s - 60.0))
                vz_dur = float(self.rng.uniform(30.0, 60.0))

                x_arr, y_arr, alt_arr, h_arr, spd_arr = self._simulate_single_track(
                    t_steps=t_steps,
                    x0=x_init,
                    y0=y_init,
                    alt0=alt_init,
                    heading0_deg=h_deg,
                    speed_kt=v_kt,
                    turn_rate_dps=turn_rate,
                    turn_start_s=turn_start,
                    turn_duration_s=turn_dur,
                    vz_fpm=vz,
                    vz_start_s=vz_start,
                    vz_duration_s=vz_dur,
                )

                for t, x_nm, y_nm, alt_ft, hdg, spd in zip(
                    t_steps, x_arr, y_arr, alt_arr, h_arr, spd_arr
                ):
                    aircraft_records.append(
                        {
                            "scenario_id": scenario_id,
                            "aircraft_id": acid,
                            "t": float(t),
                            "x_nm": round(float(x_nm), 4),
                            "y_nm": round(float(y_nm), 4),
                            "alt_ft": round(float(alt_ft), 1),
                            "heading_deg": round(float(hdg), 2),
                            "speed_kt": round(float(spd), 2),
                        }
                    )

            scenario_df = pd.DataFrame(aircraft_records)

            # Ground truth verification using exact geometry checks
            actual_has_conflict = has_scenario_conflict(
                scenario_df,
                lateral_min_nm=cfg.lateral_min_nm,
                vertical_min_ft=cfg.vertical_min_ft,
            )

            if (inject_conflict and actual_has_conflict) or (not inject_conflict and not actual_has_conflict):
                if cfg.dataset_variant == "hard":
                    n_records = len(scenario_df)
                    noise_x = self.rng.normal(0.0, cfg.noise_sigma_horizontal_nm, size=n_records)
                    noise_y = self.rng.normal(0.0, cfg.noise_sigma_horizontal_nm, size=n_records)
                    noise_alt = self.rng.normal(0.0, cfg.noise_sigma_vertical_ft, size=n_records)
                    scenario_df = scenario_df.copy()
                    scenario_df["x_obs_nm"] = np.round(scenario_df["x_nm"].to_numpy(dtype=np.float64) + noise_x, 4)
                    scenario_df["y_obs_nm"] = np.round(scenario_df["y_nm"].to_numpy(dtype=np.float64) + noise_y, 4)
                    scenario_df["alt_obs_ft"] = np.round(scenario_df["alt_ft"].to_numpy(dtype=np.float64) + noise_alt, 1)

                if not inject_conflict and should_near_miss:
                    ac0 = scenario_df[scenario_df["aircraft_id"] == "AC_0"].sort_values("t")
                    ac1 = scenario_df[scenario_df["aircraft_id"] == "AC_1"].sort_values("t")
                    dx = ac0["x_nm"].to_numpy() - ac1["x_nm"].to_numpy()
                    dy = ac0["y_nm"].to_numpy() - ac1["y_nm"].to_numpy()
                    dz = np.abs(ac0["alt_ft"].to_numpy() - ac1["alt_ft"].to_numpy())
                    d_lat = np.hypot(dx, dy)
                    closest_lat = round(float(np.min(d_lat)), 4)
                    closest_vert = round(float(np.min(dz)), 1)
                else:
                    closest_lat = None
                    closest_vert = None

                meta = {
                    "injected_conflict": bool(inject_conflict),
                    "maneuvering": bool(should_maneuver) if inject_conflict else False,
                    "maneuver_type": chosen_mtype if (inject_conflict and should_maneuver) else None,
                    "near_miss": bool(should_near_miss) if not inject_conflict else False,
                    "near_miss_closest_lateral_nm": closest_lat,
                    "near_miss_closest_vertical_ft": closest_vert,
                }
                if return_metadata:
                    return scenario_df, meta
                return scenario_df

        # Fallback return after retries
        if cfg.dataset_variant == "hard":
            n_records = len(scenario_df)
            noise_x = self.rng.normal(0.0, cfg.noise_sigma_horizontal_nm, size=n_records)
            noise_y = self.rng.normal(0.0, cfg.noise_sigma_horizontal_nm, size=n_records)
            noise_alt = self.rng.normal(0.0, cfg.noise_sigma_vertical_ft, size=n_records)
            scenario_df = scenario_df.copy()
            scenario_df["x_obs_nm"] = np.round(scenario_df["x_nm"].to_numpy(dtype=np.float64) + noise_x, 4)
            scenario_df["y_obs_nm"] = np.round(scenario_df["y_nm"].to_numpy(dtype=np.float64) + noise_y, 4)
            scenario_df["alt_obs_ft"] = np.round(scenario_df["alt_ft"].to_numpy(dtype=np.float64) + noise_alt, 1)

        if not inject_conflict and should_near_miss:
            ac0 = scenario_df[scenario_df["aircraft_id"] == "AC_0"].sort_values("t")
            ac1 = scenario_df[scenario_df["aircraft_id"] == "AC_1"].sort_values("t")
            dx = ac0["x_nm"].to_numpy() - ac1["x_nm"].to_numpy()
            dy = ac0["y_nm"].to_numpy() - ac1["y_nm"].to_numpy()
            dz = np.abs(ac0["alt_ft"].to_numpy() - ac1["alt_ft"].to_numpy())
            d_lat = np.hypot(dx, dy)
            closest_lat = round(float(np.min(d_lat)), 4)
            closest_vert = round(float(np.min(dz)), 1)
        else:
            closest_lat = None
            closest_vert = None

        meta = {
            "injected_conflict": bool(inject_conflict),
            "maneuvering": bool(should_maneuver) if inject_conflict else False,
            "maneuver_type": chosen_mtype if (inject_conflict and should_maneuver) else None,
            "near_miss": bool(should_near_miss) if not inject_conflict else False,
            "near_miss_closest_lateral_nm": closest_lat,
            "near_miss_closest_vertical_ft": closest_vert,
        }
        if return_metadata:
            return scenario_df, meta
        return scenario_df


def generate_synthetic_dataset(
    num_scenarios: int = 500,
    conflict_probability: float = 0.30,
    config: TrajectoryConfig | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate N synthetic scenarios and save to Parquet files.

    Parameters
    ----------
    num_scenarios : int, default 500
        Total number of scenarios to generate.
    conflict_probability : float, default 0.30
        Target fraction of scenarios with deliberate conflict injection (~30%).
    config : TrajectoryConfig, optional
        Simulation configuration. If None, default TrajectoryConfig is used.
    output_dir : str or Path, optional
        Target directory to save parquet files. Defaults to config.output_dir.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary of dataset generation summary metrics.
    """
    cfg = config or TrajectoryConfig()
    out_path = Path(output_dir or cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generator = ScenarioGenerator(config=cfg, seed=seed)

    # Determine scenario plan to match quotas
    num_conflicts = int(round(num_scenarios * conflict_probability))
    num_clean = num_scenarios - num_conflicts
    rng = np.random.default_rng(seed)

    if cfg.dataset_variant == "hard":
        num_maneuver = int(round(num_conflicts * cfg.maneuver_conflict_prob))
        num_climb = int(round(num_maneuver / 3.0))
        num_turn = num_maneuver - num_climb
        num_non_maneuver = num_conflicts - num_maneuver

        num_near_miss = int(round(num_scenarios * cfg.near_miss_scenario_prob))
        num_near_miss = min(num_near_miss, num_clean)
        num_std_clean = num_clean - num_near_miss

        scenario_plans: list[dict[str, Any]] = (
            [{"inject_conflict": True, "is_maneuvering": True, "maneuver_type": "turn", "is_near_miss": False}] * num_turn
            + [{"inject_conflict": True, "is_maneuvering": True, "maneuver_type": "climb_level_off", "is_near_miss": False}] * num_climb
            + [{"inject_conflict": True, "is_maneuvering": False, "maneuver_type": None, "is_near_miss": False}] * num_non_maneuver
            + [{"inject_conflict": False, "is_maneuvering": False, "maneuver_type": None, "is_near_miss": True}] * num_near_miss
            + [{"inject_conflict": False, "is_maneuvering": False, "maneuver_type": None, "is_near_miss": False}] * num_std_clean
        )
    else:
        scenario_plans = (
            [{"inject_conflict": True, "is_maneuvering": False, "maneuver_type": None, "is_near_miss": False}] * num_conflicts
            + [{"inject_conflict": False, "is_maneuvering": False, "maneuver_type": None, "is_near_miss": False}] * num_clean
        )

    rng.shuffle(scenario_plans)

    total_rows = 0
    actual_conflict_count = 0
    actual_clean_count = 0
    manifest: dict[str, Any] = {}
    metadata_map: dict[str, Any] = {}

    for idx, plan in enumerate(scenario_plans):
        scenario_id = f"scenario_{idx:04d}"
        scenario_df, meta = generator.generate_scenario(
            scenario_id=scenario_id,
            inject_conflict=plan["inject_conflict"],
            return_metadata=True,
            is_maneuvering=plan.get("is_maneuvering"),
            maneuver_type=plan.get("maneuver_type"),
            is_near_miss=plan.get("is_near_miss"),
        )

        # Confirm ground truth conflict status
        conflicts = compute_scenario_conflicts(
            scenario_df,
            lateral_min_nm=cfg.lateral_min_nm,
            vertical_min_ft=cfg.vertical_min_ft,
        )
        if len(conflicts) > 0:
            actual_conflict_count += 1
        else:
            actual_clean_count += 1

        manifest[scenario_id] = {
            "injected": bool(plan["inject_conflict"]),
            "label": bool(len(conflicts) > 0),
        }
        metadata_map[scenario_id] = meta

        file_path = out_path / f"{scenario_id}.parquet"
        table = pa.Table.from_pandas(scenario_df)
        custom_meta = {
            b"injected": b"true" if plan["inject_conflict"] else b"false",
            b"maneuvering": b"true" if meta["maneuvering"] else b"false",
            b"near_miss": b"true" if meta["near_miss"] else b"false",
        }
        merged_meta = {**(table.schema.metadata or {}), **custom_meta}
        table = table.replace_schema_metadata(merged_meta)
        pq.write_table(table, file_path)
        total_rows += len(scenario_df)

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    meta_file = out_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=2)

    actual_conflict_fraction = (
        float(actual_conflict_count) / float(num_scenarios) if num_scenarios > 0 else 0.0
    )

    summary: dict[str, Any] = {
        "scenario_count": num_scenarios,
        "conflict_scenarios": actual_conflict_count,
        "clean_scenarios": actual_clean_count,
        "conflict_fraction": actual_conflict_fraction,
        "rows_written": total_rows,
        "parquet_files": num_scenarios,
        "metadata_file": str(meta_file.resolve()),
        "output_directory": str(out_path.resolve()),
    }

    if cfg.dataset_variant == "hard":
        num_maneuver = sum(1 for m in metadata_map.values() if m["maneuvering"])
        num_turn = sum(1 for m in metadata_map.values() if m.get("maneuver_type") == "turn")
        num_climb = sum(1 for m in metadata_map.values() if m.get("maneuver_type") == "climb_level_off")
        num_near_miss = sum(1 for m in metadata_map.values() if m["near_miss"])

        nm_lats = [
            m["near_miss_closest_lateral_nm"]
            for m in metadata_map.values()
            if m["near_miss"] and m["near_miss_closest_lateral_nm"] is not None
        ]
        nm_verts = [
            m["near_miss_closest_vertical_ft"]
            for m in metadata_map.values()
            if m["near_miss"] and m["near_miss_closest_vertical_ft"] is not None
        ]

        nm_lat_min = float(np.min(nm_lats)) if nm_lats else 0.0
        nm_lat_med = float(np.median(nm_lats)) if nm_lats else 0.0
        nm_lat_max = float(np.max(nm_lats)) if nm_lats else 0.0

        nm_vert_min = float(np.min(nm_verts)) if nm_verts else 0.0
        nm_vert_med = float(np.median(nm_verts)) if nm_verts else 0.0
        nm_vert_max = float(np.max(nm_verts)) if nm_verts else 0.0

        summary["maneuvering_conflicts"] = num_maneuver
        summary["turn_conflicts"] = num_turn
        summary["climb_level_off_conflicts"] = num_climb
        summary["near_miss_scenarios"] = num_near_miss
        summary["near_miss_fraction_all"] = (
            float(num_near_miss) / float(num_scenarios) if num_scenarios > 0 else 0.0
        )
        summary["near_miss_fraction_clean"] = (
            float(num_near_miss) / float(actual_clean_count) if actual_clean_count > 0 else 0.0
        )
        summary["near_miss_lateral_min"] = nm_lat_min
        summary["near_miss_lateral_median"] = nm_lat_med
        summary["near_miss_lateral_max"] = nm_lat_max
        summary["near_miss_vertical_min"] = nm_vert_min
        summary["near_miss_vertical_median"] = nm_vert_med
        summary["near_miss_vertical_max"] = nm_vert_max

        print("=" * 70)
        print("SYNTHETIC TRAJECTORY DATASET GENERATION SUMMARY (HARD VARIANT)")
        print("=" * 70)
        print(f"Scenario count            : {summary['scenario_count']}")
        print(f"Conflict scenarios        : {summary['conflict_scenarios']}")
        print(f"Clean scenarios           : {summary['clean_scenarios']}")
        print(f"Conflict fraction         : {summary['conflict_fraction']:.4f}")
        print(
            f"Maneuvering conflicts     : {summary['maneuvering_conflicts']} "
            f"(turns: {summary['turn_conflicts']}, climb/level-off: {summary['climb_level_off_conflicts']})"
        )
        print(
            "Maneuver implementations  : Implemented climb/level-off maneuvers "
            "(AC_0 climbs/descends and levels off before rendezvous) and turn maneuvers"
        )
        print(
            f"Near-miss scenarios       : {summary['near_miss_scenarios']} "
            f"({summary['near_miss_fraction_all']*100:.1f}% of all, {summary['near_miss_fraction_clean']*100:.1f}% of clean)"
        )
        print(
            f"Near-miss closest lateral : min={summary['near_miss_lateral_min']:.2f} NM, "
            f"median={summary['near_miss_lateral_median']:.2f} NM, max={summary['near_miss_lateral_max']:.2f} NM"
        )
        print(
            f"Near-miss closest vertical: min={summary['near_miss_vertical_min']:.1f} ft, "
            f"median={summary['near_miss_vertical_median']:.1f} ft, max={summary['near_miss_vertical_max']:.1f} ft"
        )
        print(f"Rows written              : {summary['rows_written']}")
        print(f"Parquet files written     : {summary['parquet_files']}")
        print(f"Metadata file written     : {summary['metadata_file']}")
        print(f"Output directory          : {summary['output_directory']}")
        print("=" * 70)
    else:
        print("=" * 70)
        print("SYNTHETIC TRAJECTORY DATASET GENERATION SUMMARY")
        print("=" * 70)
        print(f"Scenario count       : {summary['scenario_count']}")
        print(f"Conflict scenarios   : {summary['conflict_scenarios']}")
        print(f"Clean scenarios      : {summary['clean_scenarios']}")
        print(f"Conflict fraction    : {summary['conflict_fraction']:.4f}")
        print(f"Rows written         : {summary['rows_written']}")
        print(f"Parquet files written: {summary['parquet_files']}")
        print(f"Output directory     : {summary['output_directory']}")
        print("=" * 70)

    return summary


def main() -> None:
    """CLI entry point for synthetic scenario generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic air-traffic scenarios with conflict injection (research-only)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="Total number of scenarios to generate (default: from config or 500)",
    )
    parser.add_argument(
        "--conflict-prob",
        type=float,
        default=None,
        help="Conflict probability (default: from config or 0.30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Parquet files (default: data/synthetic)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["easy", "hard", "hard_large"],
        help="Dataset variant to generate (default: easy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config)",
    )

    args = parser.parse_args()

    cfg = TrajectoryConfig.from_yaml(args.config, variant=args.variant)
    num_scenarios = args.num_scenarios if args.num_scenarios is not None else cfg.default_num_scenarios
    conflict_prob = args.conflict_prob if args.conflict_prob is not None else cfg.conflict_probability
    out_dir = args.output_dir if args.output_dir is not None else cfg.output_dir
    seed = args.seed if args.seed is not None else cfg.seed

    generate_synthetic_dataset(
        num_scenarios=num_scenarios,
        conflict_probability=conflict_prob,
        config=cfg,
        output_dir=out_dir,
        seed=seed,
    )


if __name__ == "__main__":
    main()
