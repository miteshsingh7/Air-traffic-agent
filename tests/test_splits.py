"""Unit tests for scenario-level dataset splits.

Verifies:
- Scenario-level 70/15/15 ratio
- Complete disjointness across train, val, and test sets
- Absence of scenario_id duplication
- Persistence and loading consistency via data/splits.json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from src.data.splits import (
    create_scenario_splits,
    load_splits,
    save_splits,
)


def test_split_disjointness_and_ratios():
    """Verify train/val/test splits are strictly disjoint and respect 70/15/15 ratio."""
    scenario_ids = [f"scenario_{i:04d}" for i in range(500)]
    splits = create_scenario_splits(
        scenario_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    # Disjointness assertions
    assert len(train.intersection(val)) == 0, "Train and Val must be disjoint"
    assert len(train.intersection(test)) == 0, "Train and Test must be disjoint"
    assert len(val.intersection(test)) == 0, "Val and Test must be disjoint"

    # Completeness
    assert len(train) + len(val) + len(test) == 500

    # Ratios
    assert len(train) == 350  # 70%
    assert len(val) == 75     # 15%
    assert len(test) == 75    # 15%


def test_split_persistence_and_reproducibility():
    """Verify saving and loading splits retains exact partitions and seed consistency."""
    scenario_ids = [f"scenario_{i:04d}" for i in range(100)]
    splits1 = create_scenario_splits(scenario_ids, seed=99)
    splits2 = create_scenario_splits(scenario_ids, seed=99)
    assert splits1 == splits2, "Splits must be deterministic with same seed"

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "splits.json"
        save_splits(splits1, json_path)
        loaded = load_splits(json_path)
        assert loaded == splits1


def test_invalid_split_inputs():
    """Verify error handling on invalid inputs or non-unique IDs."""
    with pytest.raises(ValueError, match="duplicate"):
        create_scenario_splits(["scen_01", "scen_01"])

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        create_scenario_splits(["scen_01", "scen_02"], train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
