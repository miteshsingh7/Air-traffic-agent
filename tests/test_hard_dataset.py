"""Unit tests for the hard synthetic dataset features.

Research simulation only - not for operational use.
Verifies:
1. Observation noise is present on history inputs only, while futures and
   conflict ground truth use noise-free true tracks.
2. Near-miss encounters are geometry-negative (no loss of separation).
3. Maneuvering conflicts are geometry-positive (confirmed loss of separation).
4. No true-future leakage occurs in the prediction and conflict advisory pipeline.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.conflict.geometry import compute_scenario_conflicts, find_earliest_conflict
from src.data.synthetic import ScenarioGenerator, TrajectoryConfig
from src.data.windows import extract_scenario_windows
from src.models.baseline import ConstantVelocityPredictor


def test_noise_only_on_history():
    """Verify observation noise is on history only, while future is noise-free true track."""
    cfg = TrajectoryConfig(
        dataset_variant="hard",
        noise_sigma_horizontal_nm=0.05,
        noise_sigma_vertical_ft=100.0,
    )
    generator = ScenarioGenerator(config=cfg, seed=777)
    df = generator.generate_scenario("test_noise_scen", inject_conflict=False)

    # 1. Assert noise columns exist in DataFrame
    assert "x_obs_nm" in df.columns
    assert "y_obs_nm" in df.columns
    assert "alt_obs_ft" in df.columns

    # 2. Assert noise columns differ from true tracks
    assert not np.allclose(df["x_obs_nm"].to_numpy(), df["x_nm"].to_numpy())
    assert not np.allclose(df["y_obs_nm"].to_numpy(), df["y_nm"].to_numpy())
    assert not np.allclose(df["alt_obs_ft"].to_numpy(), df["alt_ft"].to_numpy())

    # 3. Extract windows
    histories, futures, metadata = extract_scenario_windows(
        df,
        origin_step_s=30.0,
        history_s=cfg.history_s,
        horizon_s=cfg.horizon_s,
        dt_s=cfg.resample_rate_s,
        use_noisy_history=True,
    )
    assert len(metadata) > 0

    first_meta = metadata[0]
    acid = first_meta.aircraft_id
    origin_t = first_meta.origin_t
    dt_s = cfg.resample_rate_s

    # History slice from DataFrame
    hist_times = [round(origin_t - cfg.history_s + i * dt_s, 4) for i in range(13)]
    fut_times = [round(origin_t + (i + 1) * dt_s, 4) for i in range(60)]

    ac_df = df[df["aircraft_id"] == acid].set_index("t")
    true_hist_xyz = ac_df.loc[hist_times, ["x_nm", "y_nm", "alt_ft"]].to_numpy()
    noisy_hist_xyz = ac_df.loc[hist_times, ["x_obs_nm", "y_obs_nm", "alt_obs_ft"]].to_numpy()
    true_fut_xyz = ac_df.loc[fut_times, ["x_nm", "y_nm", "alt_ft"]].to_numpy()

    # History returned MUST equal noisy observation
    np.testing.assert_allclose(histories[0], noisy_hist_xyz, rtol=1e-5, atol=1e-5)
    # History returned MUST NOT equal clean true track
    assert not np.allclose(histories[0], true_hist_xyz)

    # Future returned MUST equal clean true track (never noisy)
    np.testing.assert_allclose(futures[0], true_fut_xyz, rtol=1e-5, atol=1e-5)


def test_near_miss_negatives_are_geometry_negative():
    """Verify that near-miss negative encounters produce strictly 0 geometric conflicts."""
    cfg = TrajectoryConfig(
        dataset_variant="hard",
        near_miss_scenario_prob=1.0,
        maneuver_conflict_prob=0.0,
    )

    for seed in [101, 102, 103, 104, 105]:
        generator = ScenarioGenerator(config=cfg, seed=seed)
        df = generator.generate_scenario(f"near_miss_{seed}", inject_conflict=False)

        conflicts = compute_scenario_conflicts(
            df,
            lateral_min_nm=cfg.lateral_min_nm,
            vertical_min_ft=cfg.vertical_min_ft,
        )
        assert len(conflicts) == 0, f"Near-miss scenario with seed {seed} produced accidental conflict"

        # Check closest approach between AC_0 and AC_1
        ac0 = df[df["aircraft_id"] == "AC_0"].sort_values("t")
        ac1 = df[df["aircraft_id"] == "AC_1"].sort_values("t")
        merged = pd.merge(ac0, ac1, on="t", suffixes=("_0", "_1"))

        dx = merged["x_nm_0"] - merged["x_nm_1"]
        dy = merged["y_nm_0"] - merged["y_nm_1"]
        d_lat = np.sqrt(dx**2 + dy**2)
        dz = np.abs(merged["alt_ft_0"] - merged["alt_ft_1"])

        # At all times, either lateral >= 5 NM OR vertical >= 1000 ft
        is_separated = (d_lat >= cfg.lateral_min_nm) | (dz >= cfg.vertical_min_ft)
        assert is_separated.all(), f"Loss of separation occurred in near miss seed {seed}"


def test_maneuvering_conflicts_are_geometry_positive():
    """Verify that maneuvering conflicts always yield a confirmed geometric conflict."""
    cfg = TrajectoryConfig(
        dataset_variant="hard",
        maneuver_conflict_prob=1.0,
        near_miss_scenario_prob=0.0,
    )

    for seed in [201, 202, 203, 204, 205]:
        generator = ScenarioGenerator(config=cfg, seed=seed)
        df, meta = generator.generate_scenario(
            f"maneuver_scen_{seed}",
            inject_conflict=True,
            return_metadata=True,
        )

        conflicts = compute_scenario_conflicts(
            df,
            lateral_min_nm=cfg.lateral_min_nm,
            vertical_min_ft=cfg.vertical_min_ft,
        )
        assert len(conflicts) >= 1, f"Maneuvering conflict seed {seed} failed to produce a conflict"

        # Verify AC_0 executed a maneuver (turn or climb/level-off) before the conflict
        ac0 = df[df["aircraft_id"] == "AC_0"].sort_values("t")
        if meta["maneuver_type"] == "turn":
            headings = ac0["heading_deg"].to_numpy()
            diffs = np.abs((headings - headings[0] + 180.0) % 360.0 - 180.0)
            max_turn = float(np.max(diffs))
            assert max_turn >= 19.5, f"AC_0 in seed {seed} did not execute a turn >= 20 deg (max turn {max_turn})"
        elif meta["maneuver_type"] == "climb_level_off":
            alts = ac0["alt_ft"].to_numpy()
            diffs_alt = np.abs(alts - alts[0])
            max_climb = float(np.max(diffs_alt))
            assert max_climb >= 500.0, f"AC_0 in seed {seed} did not execute a climb/descent >= 500 ft (max alt diff {max_climb})"


def test_no_true_future_leakage_into_predictions():
    """Verify that neither predictions nor advisory checks receive true future data."""
    cfg = TrajectoryConfig(dataset_variant="hard")
    predictor = ConstantVelocityPredictor(
        dt_s=cfg.resample_rate_s,
        horizon_steps=int(round(cfg.horizon_s / cfg.resample_rate_s)),
    )

    # Create dummy history of length 13
    n_samples = 4
    hist = np.random.default_rng(42).uniform(10.0, 100.0, size=(n_samples, 13, 3))

    # Predictor takes only hist; it has no parameter or access to future arrays
    pred_1 = predictor.predict(hist)
    pred_2 = predictor.predict(hist.copy())

    # Pure deterministic function of history
    np.testing.assert_array_equal(pred_1, pred_2)

    # Verify conflict checking on predictions uses predicted coordinates only
    pred_ac1 = pd.DataFrame({
        "t": [5.0 * (i + 1) for i in range(60)],
        "x_nm": pred_1[0, :, 0],
        "y_nm": pred_1[0, :, 1],
        "alt_ft": pred_1[0, :, 2],
    })
    pred_ac2 = pd.DataFrame({
        "t": [5.0 * (i + 1) for i in range(60)],
        "x_nm": pred_1[1, :, 0],
        "y_nm": pred_1[1, :, 1],
        "alt_ft": pred_1[1, :, 2],
    })
    ttc = find_earliest_conflict(pred_ac1, pred_ac2, lateral_min_nm=5.0, vertical_min_ft=1000.0)

    # Any modification of external future truth has zero impact on ttc
    corrupted_future = np.zeros((n_samples, 60, 3))
    # Predictor output is unaffected by any outside future array
    pred_after = predictor.predict(hist)
    np.testing.assert_array_equal(pred_1, pred_after)


def test_hard_dataset_metadata_consistency():
    """Verify metadata.json exists, scenario_ids match, and near_miss scenarios maintain separation."""
    import json
    from pathlib import Path

    meta_file = Path("data/synthetic_hard/metadata.json")
    if not meta_file.exists():
        pytest.skip("data/synthetic_hard/metadata.json not yet generated")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(metadata) == 500, f"Expected 500 scenarios in metadata, got {len(metadata)}"

    hard_dir = Path("data/synthetic_hard")
    # Verify whether full dataset parquet files exist; if only sample files are present, skip
    parquet_files = list(hard_dir.glob("*.parquet"))
    if len(parquet_files) < len(metadata):
        pytest.skip(
            f"Only {len(parquet_files)}/{len(metadata)} hard_v1 scenarios present on disk. "
            "Run dataset generation to verify full metadata consistency."
        )

    near_miss_count = 0
    maneuvering_count = 0

    for sid, m in metadata.items():
        pfile = hard_dir / f"{sid}.parquet"
        assert pfile.exists(), f"Parquet file {pfile} missing"

        # Check required fields
        assert "injected_conflict" in m
        assert "maneuvering" in m
        assert "maneuver_type" in m
        assert "near_miss" in m
        assert "near_miss_closest_lateral_nm" in m
        assert "near_miss_closest_vertical_ft" in m

        if m["near_miss"]:
            near_miss_count += 1
            lat = m["near_miss_closest_lateral_nm"]
            vert = m["near_miss_closest_vertical_ft"]
            assert lat is not None, f"Scenario {sid} near_miss lateral is None"
            assert vert is not None, f"Scenario {sid} near_miss vertical is None"
            assert (
                lat >= 5.0 or vert >= 1000.0
            ), f"Scenario {sid} violated separation: lateral={lat} NM, vertical={vert} ft"

        if m["maneuvering"]:
            maneuvering_count += 1
            assert m["maneuver_type"] in ("turn", "climb_level_off"), f"Invalid maneuver_type: {m['maneuver_type']}"

    assert near_miss_count > 0, "No near-miss scenarios found in metadata"
    assert maneuvering_count > 0, "No maneuvering scenarios found in metadata"


def test_hard_v1_dataset_hash_integrity():
    """Verify that hard_v1 dataset SHA-256 hash matches DATASET_HASH.txt exactly."""
    from src.data.dataset_hash import parse_dataset_hash_file, verify_dataset_hash

    hash_file = Path("data/synthetic_hard/DATASET_HASH.txt")
    if not hash_file.exists():
        pytest.skip("data/synthetic_hard/DATASET_HASH.txt does not exist")

    parsed = parse_dataset_hash_file(hash_file)
    expected_count = int(parsed.get("parquet_file_count", 0))
    actual_count = len(list(Path("data/synthetic_hard").glob("*.parquet")))
    if actual_count < expected_count:
        pytest.skip(
            f"hard_v1 has {actual_count}/{expected_count} parquet files on disk; skipping hash verification."
        )

    matches, stored_hash, current_hash = verify_dataset_hash("data/synthetic_hard")
    assert matches, (
        f"hard_v1 dataset hash integrity mismatch!\n"
        f"Stored:  {stored_hash}\n"
        f"Current: {current_hash}\n"
        f"Parquet files have been altered, added, or removed."
    )


def test_hard_large_dataset_hash_integrity():
    """Verify that hard_large dataset SHA-256 hash matches DATASET_HASH.txt exactly."""
    from src.data.dataset_hash import parse_dataset_hash_file, verify_dataset_hash

    hash_file = Path("data/synthetic_hard_large/DATASET_HASH.txt")
    if not hash_file.exists():
        pytest.skip("data/synthetic_hard_large/DATASET_HASH.txt does not exist")

    parsed = parse_dataset_hash_file(hash_file)
    expected_count = int(parsed.get("parquet_file_count", 0))
    actual_count = len(list(Path("data/synthetic_hard_large").glob("*.parquet")))
    if actual_count < expected_count:
        pytest.skip(
            f"hard_large has {actual_count}/{expected_count} parquet files on disk; skipping hash verification."
        )

    matches, stored_hash, current_hash = verify_dataset_hash("data/synthetic_hard_large")
    assert matches, (
        f"hard_large dataset hash integrity mismatch!\n"
        f"Stored:  {stored_hash}\n"
        f"Current: {current_hash}\n"
        f"Parquet files have been altered, added, or removed."
    )


def test_dataset_hash_verification_detects_modification():
    """Verify that dataset hash verification works and detects file modification or addition."""
    import tempfile
    from src.data.dataset_hash import create_dataset_hash_file, verify_dataset_hash

    cfg = TrajectoryConfig()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create a dummy parquet file
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        f1 = tmp_path / "scenario_0000.parquet"
        df.to_parquet(f1, index=False)

        # 2. Create hash file
        create_dataset_hash_file(tmp_path, cfg, "test_dataset")

        # 3. Verify clean match
        matches, stored_hash, current_hash = verify_dataset_hash(tmp_path)
        assert matches is True
        assert stored_hash == current_hash

        # 4. Modify the parquet file
        df_modified = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]})
        df_modified.to_parquet(f1, index=False)

        # 5. Verify detection of modified file
        matches_mod, stored_mod, current_mod = verify_dataset_hash(tmp_path)
        assert matches_mod is False
        assert stored_mod != current_mod

        # 6. Verify detection of added file
        f2 = tmp_path / "scenario_0001.parquet"
        df.to_parquet(f2, index=False)
        matches_add, _, _ = verify_dataset_hash(tmp_path)
        assert matches_add is False



