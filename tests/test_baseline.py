"""Unit tests for constant-velocity baseline predictor and trajectory error metrics.

Verifies:
- Constant-velocity exactness on straight-line tracks (error ~ 0)
- Single and batch prediction support
- Position error metric computation at horizons 30, 60, 120, 180, 240, 300 s
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import evaluate_position_errors
from src.models.baseline import ConstantVelocityPredictor


def test_constant_velocity_exactness_on_straight_track():
    """Verify that constant velocity predictor has ~0 error on a straight-line track."""
    dt_s = 5.0
    history_steps = 13  # 60s history: 12 steps + origin
    horizon_steps = 60  # 300s horizon

    # Arbitrary initial state and constant velocity
    x0, y0, z0 = 50.0, 120.0, 31000.0
    vx, vy, vz = 0.10, -0.06, 15.0  # NM/s and ft/s

    # History: t in [-60, -55, ..., 0]
    hist_times = np.arange(-60.0, dt_s, dt_s)  # 13 points
    history = np.zeros((history_steps, 3), dtype=np.float64)
    history[:, 0] = x0 + vx * hist_times
    history[:, 1] = y0 + vy * hist_times
    history[:, 2] = z0 + vz * hist_times

    # Future: t in [5, 10, ..., 300]
    fut_times = np.arange(5.0, 305.0, dt_s)  # 60 points
    future_true = np.zeros((horizon_steps, 3), dtype=np.float64)
    future_true[:, 0] = x0 + vx * fut_times
    future_true[:, 1] = y0 + vy * fut_times
    future_true[:, 2] = z0 + vz * fut_times

    model = ConstantVelocityPredictor(dt_s=dt_s, horizon_steps=horizon_steps)
    future_pred = model.predict(history)

    # Prediction must match ground truth with machine precision
    np.testing.assert_allclose(
        future_pred,
        future_true,
        atol=1e-8,
        err_msg="Constant-velocity predictor failed exactness on straight-line track",
    )

    # Position error metrics must be ~0 at all horizons
    batch_pred = future_pred[np.newaxis, :, :]
    batch_true = future_true[np.newaxis, :, :]
    metrics = evaluate_position_errors(batch_pred, batch_true, dt_s=dt_s)

    for m in metrics:
        assert m.horizontal_rmse_nm == pytest.approx(0.0, abs=1e-6)
        assert m.vertical_rmse_ft == pytest.approx(0.0, abs=1e-6)


def test_batch_prediction_shape():
    """Verify batch prediction handling and shapes."""
    batch_size = 10
    history = np.random.randn(batch_size, 13, 3)
    model = ConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)
    pred = model.predict(history)
    assert pred.shape == (batch_size, 60, 3)


def test_smoothed_cv_exactness_on_straight_track():
    """Verify that SmoothedConstantVelocityPredictor is exact (~0 error) on a straight track."""
    from src.models.baseline import SmoothedConstantVelocityPredictor

    dt_s = 5.0
    history_steps = 13
    horizon_steps = 60

    x0, y0, z0 = 60.0, 110.0, 28000.0
    vx, vy, vz = 0.08, -0.05, 12.0

    hist_times = np.arange(-60.0, dt_s, dt_s)
    history = np.zeros((history_steps, 3), dtype=np.float64)
    history[:, 0] = x0 + vx * hist_times
    history[:, 1] = y0 + vy * hist_times
    history[:, 2] = z0 + vz * hist_times

    fut_times = np.arange(5.0, 305.0, dt_s)
    future_true = np.zeros((horizon_steps, 3), dtype=np.float64)
    future_true[:, 0] = x0 + vx * fut_times
    future_true[:, 1] = y0 + vy * fut_times
    future_true[:, 2] = z0 + vz * fut_times

    model = SmoothedConstantVelocityPredictor(dt_s=dt_s, horizon_steps=horizon_steps)
    future_pred = model.predict(history)

    np.testing.assert_allclose(
        future_pred,
        future_true,
        atol=1e-8,
        err_msg="SmoothedConstantVelocityPredictor failed exactness on straight-line track",
    )


def test_smoothed_cv_lower_rmse_than_3step_with_noise():
    """Verify that cv_smoothed has lower position RMSE than cv_3step on noisy straight track."""
    from src.models.baseline import ConstantVelocityPredictor, SmoothedConstantVelocityPredictor

    dt_s = 5.0
    history_steps = 13
    horizon_steps = 60

    rng = np.random.default_rng(2026)
    n_trials = 200

    x0, y0, z0 = 100.0, 100.0, 30000.0
    vx, vy, vz = 0.10, -0.05, 10.0

    hist_times = np.arange(-60.0, dt_s, dt_s)
    clean_hist = np.zeros((history_steps, 3), dtype=np.float64)
    clean_hist[:, 0] = x0 + vx * hist_times
    clean_hist[:, 1] = y0 + vy * hist_times
    clean_hist[:, 2] = z0 + vz * hist_times

    fut_times = np.arange(5.0, 305.0, dt_s)
    clean_fut = np.zeros((horizon_steps, 3), dtype=np.float64)
    clean_fut[:, 0] = x0 + vx * fut_times
    clean_fut[:, 1] = y0 + vy * fut_times
    clean_fut[:, 2] = z0 + vz * fut_times

    # Add Gaussian observation noise to histories across multiple trials
    noise_sigma_h = 0.05  # NM
    noise_sigma_v = 80.0  # ft

    batch_hist = np.tile(clean_hist[np.newaxis, :, :], (n_trials, 1, 1)).copy()
    batch_hist[:, :, 0] += rng.normal(0.0, noise_sigma_h, size=(n_trials, history_steps))
    batch_hist[:, :, 1] += rng.normal(0.0, noise_sigma_h, size=(n_trials, history_steps))
    batch_hist[:, :, 2] += rng.normal(0.0, noise_sigma_v, size=(n_trials, history_steps))

    model_3step = ConstantVelocityPredictor(dt_s=dt_s, horizon_steps=horizon_steps)
    model_smoothed = SmoothedConstantVelocityPredictor(dt_s=dt_s, horizon_steps=horizon_steps)

    pred_3step = model_3step.predict(batch_hist)
    pred_smoothed = model_smoothed.predict(batch_hist)

    # Position RMSE at 300s horizon (index -1)
    err_3step_h = np.sqrt(
        (pred_3step[:, -1, 0] - clean_fut[-1, 0]) ** 2
        + (pred_3step[:, -1, 1] - clean_fut[-1, 1]) ** 2
    )
    err_smoothed_h = np.sqrt(
        (pred_smoothed[:, -1, 0] - clean_fut[-1, 0]) ** 2
        + (pred_smoothed[:, -1, 1] - clean_fut[-1, 1]) ** 2
    )

    err_3step_v = np.abs(pred_3step[:, -1, 2] - clean_fut[-1, 2])
    err_smoothed_v = np.abs(pred_smoothed[:, -1, 2] - clean_fut[-1, 2])

    rmse_3step_h = float(np.sqrt(np.mean(err_3step_h**2)))
    rmse_smoothed_h = float(np.sqrt(np.mean(err_smoothed_h**2)))

    rmse_3step_v = float(np.sqrt(np.mean(err_3step_v**2)))
    rmse_smoothed_v = float(np.sqrt(np.mean(err_smoothed_v**2)))

    # Assert smoothed predictor significantly reduces horizontal and vertical RMSE
    assert rmse_smoothed_h < rmse_3step_h, f"Smoothed ({rmse_smoothed_h:.4f}) not lower than 3-step ({rmse_3step_h:.4f})"
    assert rmse_smoothed_v < rmse_3step_v, f"Smoothed ({rmse_smoothed_v:.1f}) not lower than 3-step ({rmse_3step_v:.1f})"

