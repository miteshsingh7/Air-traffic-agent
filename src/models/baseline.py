"""Constant-velocity baseline trajectory predictor.

Research simulation only - not for operational use.
Estimates velocity in x, y, and altitude from the last 3 history steps
and extrapolates linearly over the prediction horizon.
"""

from __future__ import annotations

import numpy as np


class ConstantVelocityPredictor:
    """Predicts future aircraft positions via constant-velocity linear extrapolation.

    Velocity in (x, y, alt) is estimated from the last 3 history steps:
        v = (pos[-1] - pos[-3]) / (2 * dt)
    and extrapolated linearly over the horizon:
        pos_pred(t0 + k * dt) = pos[-1] + v * (k * dt)
    """

    def __init__(self, dt_s: float = 5.0, horizon_steps: int = 60) -> None:
        """Initialize constant velocity predictor.

        Parameters
        ----------
        dt_s : float, default 5.0
            Time step duration in seconds.
        horizon_steps : int, default 60
            Number of future prediction steps (60 steps * 5s = 300s horizon).
        """
        if dt_s <= 0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")

        self.dt_s = dt_s
        self.horizon_steps = horizon_steps

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict future trajectory from history observation.

        Parameters
        ----------
        history : np.ndarray
            History positions of shape (steps, 3) or (batch, steps, 3)
            with columns [x_nm, y_nm, alt_ft]. Must have at least 3 steps.

        Returns
        -------
        np.ndarray
            Predicted future positions of shape (horizon_steps, 3)
            or (batch, horizon_steps, 3).
        """
        if history.ndim == 2:
            return self._predict_single(history)
        elif history.ndim == 3:
            return self._predict_batch(history)
        else:
            raise ValueError(f"Expected history ndim 2 or 3, got shape {history.shape}")

    def _predict_single(self, history: np.ndarray) -> np.ndarray:
        """Single trajectory prediction."""
        if history.shape[0] < 3:
            raise ValueError(f"History must contain at least 3 steps, got {history.shape[0]}")
        if history.shape[1] != 3:
            raise ValueError(f"History features must be 3 [x, y, alt], got {history.shape[1]}")

        # Velocity from last 3 steps: pos[-1] and pos[-3] span 2 * dt_s
        dt_span = 2.0 * self.dt_s
        velocity = (history[-1] - history[-3]) / dt_span  # [v_x, v_y, v_z] per second
        origin_pos = history[-1]  # position at t_origin

        # Time offsets: [dt, 2*dt, ..., horizon_steps*dt]
        time_offsets = np.arange(1, self.horizon_steps + 1, dtype=np.float64) * self.dt_s  # (H,)
        # Broadcast: (H, 1) * (1, 3) -> (H, 3)
        future_pred = origin_pos[np.newaxis, :] + time_offsets[:, np.newaxis] * velocity[np.newaxis, :]
        return future_pred

    def _predict_batch(self, history: np.ndarray) -> np.ndarray:
        """Batch trajectory prediction."""
        if history.shape[1] < 3:
            raise ValueError(f"History must contain at least 3 steps, got {history.shape[1]}")
        if history.shape[2] != 3:
            raise ValueError(f"History features must be 3 [x, y, alt], got {history.shape[2]}")

        dt_span = 2.0 * self.dt_s
        velocity = (history[:, -1, :] - history[:, -3, :]) / dt_span  # (N, 3)
        origin_pos = history[:, -1, :]  # (N, 3)

        time_offsets = np.arange(1, self.horizon_steps + 1, dtype=np.float64) * self.dt_s  # (H,)
        # (N, 1, 3) + (1, H, 1) * (N, 1, 3) -> (N, H, 3)
        future_pred = (
            origin_pos[:, np.newaxis, :]
            + time_offsets[np.newaxis, :, np.newaxis] * velocity[:, np.newaxis, :]
        )
        return future_pred


class SmoothedConstantVelocityPredictor:
    """Predicts future aircraft positions via least-squares linear fit over full history.

    Fits a straight line over all available history steps (e.g., 13 steps):
        pos_fit(k) = a + b * k
    extrapolating from the fitted current position (at step K-1) and fitted velocity:
        pos_pred(t0 + h * dt) = pos_fit(K-1) + b * h
    This filters out observation noise while remaining strictly exact on straight tracks.
    """

    def __init__(self, dt_s: float = 5.0, horizon_steps: int = 60) -> None:
        if dt_s <= 0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")

        self.dt_s = dt_s
        self.horizon_steps = horizon_steps

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict future trajectory using full-history least-squares fit."""
        if history.ndim == 2:
            return self._predict_batch(history[np.newaxis, :, :])[0]
        elif history.ndim == 3:
            return self._predict_batch(history)
        else:
            raise ValueError(f"Expected history ndim 2 or 3, got shape {history.shape}")

    def _predict_batch(self, history: np.ndarray) -> np.ndarray:
        """Vectorized batch least-squares prediction."""
        n_samples, n_steps, n_feats = history.shape
        if n_steps < 3:
            raise ValueError(f"History must contain at least 3 steps, got {n_steps}")
        if n_feats != 3:
            raise ValueError(f"History features must be 3 [x, y, alt], got {n_feats}")

        # Indices k = 0, 1, ..., n_steps - 1
        k_indices = np.arange(n_steps, dtype=np.float64)
        k_mean = (n_steps - 1.0) / 2.0
        k_centered = k_indices - k_mean  # Shape (K,)
        denom = float(np.sum(k_centered**2))

        # Mean across time dimension: shape (N, 3)
        y_mean = np.mean(history, axis=1)

        # Least-squares slope per step (displacement per dt_s step): shape (N, 3)
        # Sum along time axis (axis 1): (N, K, 3) * (1, K, 1) -> (N, 3)
        slope = np.sum(history * k_centered[np.newaxis, :, np.newaxis], axis=1) / denom

        # Fitted current position at the latest history step (k = n_steps - 1): shape (N, 3)
        current_k_offset = (n_steps - 1.0) - k_mean
        pos_current_fit = y_mean + slope * current_k_offset

        # Extrapolate over future steps h = 1, 2, ..., horizon_steps
        h_offsets = np.arange(1, self.horizon_steps + 1, dtype=np.float64)  # Shape (H,)
        # (N, 1, 3) + (1, H, 1) * (N, 1, 3) -> (N, H, 3)
        future_pred = (
            pos_current_fit[:, np.newaxis, :]
            + h_offsets[np.newaxis, :, np.newaxis] * slope[:, np.newaxis, :]
        )
        return future_pred


# Model registry aliases
cv_3step = ConstantVelocityPredictor
cv_smoothed = SmoothedConstantVelocityPredictor
