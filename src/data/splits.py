"""Scenario-level dataset splitting for train, validation, and test sets.

Research simulation only - not for operational use.
Splits strictly at the scenario level (never by window or aircraft)
to avoid temporal and multi-aircraft data leakage across splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import yaml


def create_scenario_splits(
    scenario_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create disjoint scenario-level train/val/test splits.

    Parameters
    ----------
    scenario_ids : Sequence[str]
        Full list of unique scenario identifiers.
    train_ratio : float, default 0.70
        Proportion of scenarios assigned to training.
    val_ratio : float, default 0.15
        Proportion of scenarios assigned to validation.
    test_ratio : float, default 0.15
        Proportion of scenarios assigned to testing.
    seed : int, default 42
        Random seed for reproducible shuffling.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys 'train', 'val', 'test' mapping to scenario IDs.
    """
    total = len(scenario_ids)
    if total == 0:
        raise ValueError("scenario_ids cannot be empty")

    unique_ids = sorted(set(scenario_ids))
    if len(unique_ids) != total:
        raise ValueError(f"scenario_ids contains duplicate IDs: {total} given, {len(unique_ids)} unique")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    # Deterministic shuffle
    rng = np.random.default_rng(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))
    # Test takes remainder to ensure all scenarios are accounted for
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    # Assert complete disjointness
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    assert not set_train.intersection(set_val), "Disjointness violated: train and val share scenario IDs"
    assert not set_train.intersection(set_test), "Disjointness violated: train and test share scenario IDs"
    assert not set_val.intersection(set_test), "Disjointness violated: val and test share scenario IDs"
    assert len(train_ids) + len(val_ids) + len(test_ids) == total, (
        f"Partition size mismatch: {len(train_ids)} + {len(val_ids)} + {len(test_ids)} != {total}"
    )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def save_splits(
    splits: Mapping[str, Sequence[str]],
    output_path: str | Path = "data/splits.json",
) -> Path:
    """Save dataset splits to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to pure Python lists for JSON serialization
    serialized = {k: list(v) for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)

    return path


def load_splits(splits_path: str | Path = "data/splits.json") -> dict[str, list[str]]:
    """Load dataset splits from a JSON file and verify disjointness."""
    path = Path(splits_path)
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_path}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, list[str]] = json.load(f)

    for required_key in ("train", "val", "test"):
        if required_key not in data:
            raise KeyError(f"Missing required split '{required_key}' in {splits_path}")

    train_set = set(data["train"])
    val_set = set(data["val"])
    test_set = set(data["test"])

    assert not train_set.intersection(val_set), "Loaded splits error: train and val share scenarios"
    assert not train_set.intersection(test_set), "Loaded splits error: train and test share scenarios"
    assert not val_set.intersection(test_set), "Loaded splits error: val and test share scenarios"

    return data


def get_or_create_splits(
    data_dir: str | Path | None = None,
    config_path: str | Path = "configs/default.yaml",
    variant: str = "easy",
    output_file: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Get existing splits from file or generate and save them using configuration."""
    cfg_path = Path(config_path)
    splits_cfg: dict[str, Any] = {}
    hard_cfg: dict[str, Any] = {}
    hard_large_cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            splits_cfg = full_cfg.get("splits", {})
            hard_cfg = full_cfg.get("hard_dataset", {})
            hard_large_cfg = full_cfg.get("hard_large_dataset", {})

    is_hard_large = variant == "hard_large" or (data_dir is not None and "hard_large" in str(data_dir))
    is_hard = variant == "hard" or (data_dir is not None and "hard" in str(data_dir) and not is_hard_large)

    if is_hard_large:
        default_dir = hard_large_cfg.get("output_dir", "data/synthetic_hard_large")
        default_out = hard_large_cfg.get("splits_file", "data/splits_hard_large.json")
        default_seed = int(hard_large_cfg.get("seed", 2042))
    elif is_hard:
        default_dir = hard_cfg.get("output_dir", "data/synthetic_hard")
        default_out = hard_cfg.get("splits_file", "data/splits_hard.json")
        default_seed = int(hard_cfg.get("seed", 1042))
    else:
        default_dir = "data/synthetic"
        default_out = splits_cfg.get("output_file", "data/splits.json")
        default_seed = int(splits_cfg.get("seed", 42))

    target_data_dir = Path(data_dir or default_dir)
    target_output_file = Path(output_file or default_out)
    target_seed = seed if seed is not None else default_seed

    if target_output_file.exists():
        return load_splits(target_output_file)

    train_ratio = float(splits_cfg.get("train_ratio", 0.70))
    val_ratio = float(splits_cfg.get("val_ratio", 0.15))
    test_ratio = float(splits_cfg.get("test_ratio", 0.15))

    parquet_files = sorted(target_data_dir.glob("scenario_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No scenario parquet files found in {target_data_dir}")

    scenario_ids = [p.stem for p in parquet_files]
    splits = create_scenario_splits(
        scenario_ids=scenario_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=target_seed,
    )
    save_splits(splits, target_output_file)
    return splits


def main() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Create or load scenario-level dataset splits (research-only)."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="easy",
        choices=["easy", "hard", "hard_large"],
        help="Dataset variant (easy, hard, or hard_large, default: easy)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing scenario parquet files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save splits JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    splits = get_or_create_splits(
        data_dir=args.data_dir,
        config_path=args.config,
        variant=args.variant,
        output_file=args.output_file,
        seed=args.seed,
    )
    print(f"Dataset splits ({args.variant}) created/loaded successfully:")
    print(f"  Train : {len(splits['train'])} scenarios")
    print(f"  Val   : {len(splits['val'])} scenarios")
    print(f"  Test  : {len(splits['test'])} scenarios")
    print(f"  Total : {len(splits['train']) + len(splits['val']) + len(splits['test'])} scenarios")


if __name__ == "__main__":
    main()
