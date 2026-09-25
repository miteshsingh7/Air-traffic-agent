"""Data pipelines, synthetic generation, splits, and windowing module."""

from __future__ import annotations


def __getattr__(name: str):
    if name in ("ScenarioGenerator", "TrajectoryConfig", "generate_synthetic_dataset"):
        from src.data import synthetic

        return getattr(synthetic, name)
    if name in ("create_scenario_splits", "load_splits", "save_splits", "get_or_create_splits"):
        from src.data import splits

        return getattr(splits, name)
    if name in ("WindowSample", "extract_scenario_windows", "extract_dataset_windows"):
        from src.data import windows

        return getattr(windows, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScenarioGenerator",
    "TrajectoryConfig",
    "generate_synthetic_dataset",
    "create_scenario_splits",
    "load_splits",
    "save_splits",
    "get_or_create_splits",
    "WindowSample",
    "extract_scenario_windows",
    "extract_dataset_windows",
]
