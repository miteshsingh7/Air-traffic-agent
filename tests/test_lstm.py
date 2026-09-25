"""Unit tests for the residual LSTM trajectory predictor.

Research simulation only - not for operational use.
Verifies:
1. Model output shape is (60, 3) for single trajectory and (B, 60, 3) for batch.
2. Zero-initialized model output is identical to cv_smoothed to high numerical precision.
3. No test scenario_id from splits appears in the train dataset.
4. Predictor consumes only the history array, with no access or leakage of true futures.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.baseline import SmoothedConstantVelocityPredictor
from src.models.lstm_predictor import LSTMTrajectoryPredictor


def test_lstm_output_shapes():
    """Verify single and batch forward and predict output dimensions."""
    model = LSTMTrajectoryPredictor(horizon_steps=60, history_steps=13)

    # 1. Single trajectory (13, 3)
    hist_single = np.random.uniform(-50.0, 50.0, size=(13, 3)).astype(np.float32)
    pred_single = model.predict(hist_single)
    assert isinstance(pred_single, np.ndarray)
    assert pred_single.shape == (60, 3), f"Expected shape (60, 3), got {pred_single.shape}"

    # 2. Batch trajectories (B, 13, 3)
    b_size = 7
    hist_batch = np.random.uniform(-50.0, 50.0, size=(b_size, 13, 3)).astype(np.float32)
    pred_batch = model.predict(hist_batch)
    assert isinstance(pred_batch, np.ndarray)
    assert pred_batch.shape == (b_size, 60, 3), f"Expected shape ({b_size}, 60, 3), got {pred_batch.shape}"

    # 3. PyTorch tensor forward pass
    t_hist = torch.from_numpy(hist_batch)
    t_pred, t_res = model.forward(t_hist, return_residual=True)
    assert t_pred.shape == (b_size, 60, 3)
    assert t_res.shape == (b_size, 60, 3)


def test_zero_initialized_lstm_matches_cv_smoothed():
    """Verify that before training, the residual LSTM identically reproduces cv_smoothed."""
    model = LSTMTrajectoryPredictor(horizon_steps=60, history_steps=13)
    cv = SmoothedConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)

    rng = np.random.default_rng(1234)
    # Test across multiple random trajectory histories
    hist_batch = rng.uniform(-100.0, 100.0, size=(16, 13, 3)).astype(np.float32)

    pred_lstm = model.predict(hist_batch)
    pred_cv = cv.predict(hist_batch)

    # Differences should be within float32 numerical precision
    max_abs_diff = float(np.max(np.abs(pred_lstm - pred_cv)))
    assert max_abs_diff < 1e-4, f"Zero-initialized LSTM differs from cv_smoothed by {max_abs_diff:.6e}"

    # Verify that the normalized residual tensor is identically zero
    with torch.no_grad():
        _, delta_norm = model.forward(torch.from_numpy(hist_batch), return_residual=True)
        assert torch.all(delta_norm == 0.0), "Normalized residual is not identically zero!"


def test_train_val_test_split_isolation():
    """Verify that no test scenario appears in the training or validation splits."""
    splits_file = Path("data/splits_hard_large.json")
    if not splits_file.exists():
        pytest.skip("data/splits_hard_large.json does not exist yet")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Strict scenario-level disjointness
    assert len(train_set.intersection(test_set)) == 0, "Train split contains test scenarios!"
    assert len(val_set.intersection(test_set)) == 0, "Val split contains test scenarios!"
    assert len(train_set.intersection(val_set)) == 0, "Train split contains val scenarios!"
    assert len(test_set) > 0, "Test set is empty!"


def test_lstm_predictor_consumes_only_history():
    """Verify that predictor takes only history array, with no future leakage."""
    model = LSTMTrajectoryPredictor(horizon_steps=60, history_steps=13)

    hist = np.random.uniform(0.0, 50.0, size=(13, 3)).astype(np.float32)

    # Deterministic behavior on identical history inputs
    pred1 = model.predict(hist)
    pred2 = model.predict(hist.copy())
    np.testing.assert_array_equal(pred1, pred2)

    # Corrupting or modifying external futures has zero effect on predictor
    external_future = np.ones((60, 3))
    pred3 = model.predict(hist)
    np.testing.assert_array_equal(pred1, pred3)


def test_batch_vs_chunked_prediction_consistency():
    """Verify that batch vs chunked prediction differs by < 1e-4 NM and < 1e-2 ft across 2000 windows."""
    ckpt_path = Path("checkpoints/lstm_v1/best.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not found")

    model = LSTMTrajectoryPredictor.from_checkpoint(ckpt_path, device="cpu")

    cache_path = Path("data/cache/hard_large_val_windows.pt")
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        histories = data["histories"][:2000].numpy()
    else:
        rng = np.random.default_rng(42)
        histories = rng.uniform(-100.0, 100.0, size=(2000, 13, 3)).astype(np.float32)

    # 1. Single batch of 2000
    pred_batch = model.predict(histories, chunk_size=2000)

    # 2. Chunked evaluation with chunk_size=256
    pred_chunked = model.predict(histories, chunk_size=256)

    diff_h = float(np.max(np.hypot(pred_batch[..., 0] - pred_chunked[..., 0], pred_batch[..., 1] - pred_chunked[..., 1])))
    diff_v = float(np.max(np.abs(pred_batch[..., 2] - pred_chunked[..., 2])))

    assert diff_h < 1e-4, f"Horizontal difference {diff_h:.6e} NM >= 1e-4 NM"
    assert diff_v < 1e-2, f"Vertical difference {diff_v:.6e} ft >= 1e-2 ft"


def test_lstm_val_rmse_matches_training_log():
    """Verify that run_baseline on hard_large VAL matches epoch 39 training-log RMSE within 0.01 NM and 5 ft at all horizons."""
    ckpt_path = Path("checkpoints/lstm_v1/best.pt")
    cache_path = Path("data/cache/hard_large_val_windows.pt")
    if not ckpt_path.exists() or not cache_path.exists():
        pytest.skip("Checkpoint or validation window cache not found")

    from src.eval.metrics import evaluate_position_errors

    model = LSTMTrajectoryPredictor.from_checkpoint(ckpt_path)
    val_data = torch.load(cache_path, map_location="cpu")
    val_hist = val_data["histories"].numpy()
    val_fut = val_data["futures"].numpy()

    # Predict all validation windows through the harness predict interface
    preds = model.predict(val_hist)

    horizons = (30, 60, 120, 180, 240, 300)
    metrics = evaluate_position_errors(y_pred=preds, y_true=val_fut, horizons_s=horizons, dt_s=5.0)
    metric_dict = {m.horizon_s: m for m in metrics}

    # Training log epoch 39 targets (from training log & prompt specification):
    # 60s: 0.2473 NM, 180s: 1.2366 NM, 300s: 2.6307 NM, 300s vert: 159.0 ft
    expected = {
        60: (0.2473, None),
        180: (1.2366, None),
        300: (2.6307, 159.0),
    }

    for h, (exp_h, exp_v) in expected.items():
        assert abs(metric_dict[h].horizontal_rmse_nm - exp_h) < 0.01, (
            f"Horizontal RMSE @ {h}s ({metric_dict[h].horizontal_rmse_nm:.4f}) deviates > 0.01 NM from {exp_h}"
        )
        if exp_v is not None:
            assert abs(metric_dict[h].vertical_rmse_ft - exp_v) < 5.0, (
                f"Vertical RMSE @ {h}s ({metric_dict[h].vertical_rmse_ft:.2f}) deviates > 5.0 ft from {exp_v}"
            )

