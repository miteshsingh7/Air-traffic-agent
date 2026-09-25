"""Conflict geometry and separation loss detection module."""

from src.conflict.geometry import (
    ConflictRecord,
    conflicts_to_dataframe,
    compute_scenario_conflicts,
    find_earliest_conflict,
    has_scenario_conflict,
)

__all__ = [
    "ConflictRecord",
    "conflicts_to_dataframe",
    "compute_scenario_conflicts",
    "find_earliest_conflict",
    "has_scenario_conflict",
]
