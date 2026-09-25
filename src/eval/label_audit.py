"""Label audit module for verifying synthetic scenarios against true geometry.

Research simulation only - not for operational use.
Audits all scenarios in a dataset by comparing the generation injection flag
against exact geometry-confirmed loss-of-separation ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.conflict.geometry import ConflictRecord, compute_scenario_conflicts
from src.data.synthetic import TrajectoryConfig


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics (min, median, max) for a numeric metric."""

    min_val: float
    median_val: float
    max_val: float

    def __str__(self) -> str:
        return f"min={self.min_val:.2f}, median={self.median_val:.2f}, max={self.max_val:.2f}"


@dataclass(frozen=True)
class AuditReport:
    """Full audit report comparing scenario injection flags to geometry truth."""

    total_scenarios: int
    injected_with_conflict: int
    injected_without_conflict: int
    clean_with_conflict: int
    clean_without_conflict: int
    total_confirmed_conflict_pairs: int
    earliest_conflict_time_dist: DistributionSummary | None
    closest_approach_dist_dist: DistributionSummary | None
    has_discrepancy: bool


def _load_injection_flags(data_dir: Path) -> dict[str, bool]:
    """Load injection flags from manifest.json, metadata.json, or parquet metadata."""
    manifest_file = data_dir / "manifest.json"
    metadata_file = data_dir / "metadata.json"
    flags: dict[str, bool] = {}

    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            flags[k] = bool(v.get("injected", False))
                        elif isinstance(v, bool):
                            flags[k] = v
        except Exception:
            flags = {}
    elif metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            flags[k] = bool(v.get("injected_conflict", v.get("injected", False)))
                        elif isinstance(v, bool):
                            flags[k] = v
        except Exception:
            flags = {}

    return flags


def run_label_audit(
    data_dir: str | Path = "data/synthetic",
    config: TrajectoryConfig | None = None,
) -> AuditReport:
    """Audit all scenarios in data_dir against exact geometry ground truth.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing synthetic scenario Parquet files.
    config : TrajectoryConfig, optional
        Simulation config containing separation minima.

    Returns
    -------
    AuditReport
        Report containing counts and distributions.
    """
    path = Path(data_dir)
    cfg = config or TrajectoryConfig.from_yaml("configs/default.yaml")

    parquet_files = sorted(path.glob("scenario_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No scenario parquet files found in {data_dir}")

    manifest_flags = _load_injection_flags(path)

    injected_with_conflict = 0
    injected_without_conflict = 0
    clean_with_conflict = 0
    clean_without_conflict = 0
    total_confirmed_conflict_pairs = 0

    all_earliest_times: list[float] = []
    all_closest_approaches: list[float] = []

    for p in parquet_files:
        scenario_id = p.stem

        # Determine injection flag: manifest -> parquet schema metadata -> fallback False
        if scenario_id in manifest_flags:
            is_injected = manifest_flags[scenario_id]
        else:
            try:
                parquet_meta = pq.read_metadata(p)
                raw_schema_meta = parquet_meta.schema.to_arrow_schema().metadata or {}
                raw_flag = raw_schema_meta.get(b"injected", b"false").decode("utf-8").lower()
                is_injected = raw_flag in ("true", "1", "yes")
            except Exception:
                is_injected = False

        scenario_df = pd.read_parquet(p)
        conflicts: list[ConflictRecord] = compute_scenario_conflicts(
            scenario_df,
            lateral_min_nm=cfg.lateral_min_nm,
            vertical_min_ft=cfg.vertical_min_ft,
        )

        has_geom_conflict = len(conflicts) > 0

        if is_injected:
            if has_geom_conflict:
                injected_with_conflict += 1
            else:
                injected_without_conflict += 1
        else:
            if has_geom_conflict:
                clean_with_conflict += 1
            else:
                clean_without_conflict += 1

        if has_geom_conflict:
            total_confirmed_conflict_pairs += len(conflicts)
            scen_earliest_t = min(c.first_conflict_time for c in conflicts)
            scen_min_lat = min(c.min_lateral_distance_nm for c in conflicts)
            all_earliest_times.append(scen_earliest_t)
            all_closest_approaches.append(scen_min_lat)

    earliest_dist = (
        DistributionSummary(
            min_val=float(np.min(all_earliest_times)),
            median_val=float(np.median(all_earliest_times)),
            max_val=float(np.max(all_earliest_times)),
        )
        if all_earliest_times
        else None
    )

    closest_dist = (
        DistributionSummary(
            min_val=float(np.min(all_closest_approaches)),
            median_val=float(np.median(all_closest_approaches)),
            max_val=float(np.max(all_closest_approaches)),
        )
        if all_closest_approaches
        else None
    )

    has_discrepancy = (injected_without_conflict > 0) or (clean_with_conflict > 0)

    report = AuditReport(
        total_scenarios=len(parquet_files),
        injected_with_conflict=injected_with_conflict,
        injected_without_conflict=injected_without_conflict,
        clean_with_conflict=clean_with_conflict,
        clean_without_conflict=clean_without_conflict,
        total_confirmed_conflict_pairs=total_confirmed_conflict_pairs,
        earliest_conflict_time_dist=earliest_dist,
        closest_approach_dist_dist=closest_dist,
        has_discrepancy=has_discrepancy,
    )

    # Print required unedited summary
    print("=" * 70)
    print("LABEL AUDIT REPORT")
    print("=" * 70)
    print(f"Total scenarios audited: {report.total_scenarios}")
    print("\nInjected scenarios:")
    print(f"  - Geometry-confirmed conflict : {report.injected_with_conflict}")
    print(f"  - Without confirmed conflict   : {report.injected_without_conflict}")
    print("\nClean scenarios:")
    print(f"  - Accidental geometry conflict : {report.clean_with_conflict}")
    print(f"  - Without conflict (clean)     : {report.clean_without_conflict}")
    print("\nConfirmed conflicts distribution:")
    if report.earliest_conflict_time_dist:
        print(f"  - Earliest conflict time (s)   : {report.earliest_conflict_time_dist}")
    else:
        print("  - Earliest conflict time (s)   : N/A (no conflicts)")
    if report.closest_approach_dist_dist:
        print(f"  - Closest-approach distance(NM): {report.closest_approach_dist_dist}")
    else:
        print("  - Closest-approach distance(NM): N/A (no conflicts)")
    print(f"\nTotal confirmed conflict pairs across dataset: {report.total_confirmed_conflict_pairs}")
    print("=" * 70)

    return report


def main() -> None:
    """CLI entry point for label audit."""
    parser = argparse.ArgumentParser(
        description="Audit scenario labels against true geometric conflict ground truth."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Path to directory containing scenario Parquet files (default: data/synthetic)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    cfg = TrajectoryConfig.from_yaml(args.config)
    run_label_audit(data_dir=args.data_dir, config=cfg)


if __name__ == "__main__":
    main()
