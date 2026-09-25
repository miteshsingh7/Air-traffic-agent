"""Evaluation metrics for trajectory prediction error.

Research simulation only - not for operational use.
Computes position error across prediction horizons:
- Horizontal RMSE in NM (Euclidean distance in x/y plane)
- Vertical RMSE in ft (absolute altitude error)
at horizons 30, 60, 120, 180, 240, 300 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class PositionErrorMetric:
    """Position error metric evaluated at a specific forecast horizon."""

    horizon_s: int
    horizontal_rmse_nm: float
    vertical_rmse_ft: float
    num_samples: int


def evaluate_position_errors(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    horizons_s: Sequence[int] = (30, 60, 120, 180, 240, 300),
    dt_s: float = 5.0,
) -> list[PositionErrorMetric]:
    """Compute horizontal and vertical RMSE at specified forecast horizons.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted positions of shape (N, horizon_steps, 3) [x_nm, y_nm, alt_ft].
    y_true : np.ndarray
        True future positions of shape (N, horizon_steps, 3) [x_nm, y_nm, alt_ft].
    horizons_s : Sequence[int], default (30, 60, 120, 180, 240, 300)
        List of forecast horizons in seconds to evaluate.
    dt_s : float, default 5.0
        Time step in seconds.

    Returns
    -------
    list[PositionErrorMetric]
        List of error metrics for each evaluated horizon.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}")
    if y_pred.ndim != 3 or y_pred.shape[2] != 3:
        raise ValueError(f"Expected shape (N, steps, 3), got {y_pred.shape}")

    n_samples, n_steps, _ = y_pred.shape
    if n_samples == 0:
        return [
            PositionErrorMetric(
                horizon_s=h,
                horizontal_rmse_nm=0.0,
                vertical_rmse_ft=0.0,
                num_samples=0,
            )
            for h in horizons_s
        ]

    results: list[PositionErrorMetric] = []

    for h in horizons_s:
        step_idx = int(round(h / dt_s)) - 1
        if not (0 <= step_idx < n_steps):
            raise IndexError(
                f"Horizon {h}s (step {step_idx}) exceeds available prediction steps ({n_steps})"
            )

        pred_step = y_pred[:, step_idx, :]
        true_step = y_true[:, step_idx, :]

        # Horizontal error (Euclidean in x/y)
        dx = pred_step[:, 0] - true_step[:, 0]
        dy = pred_step[:, 1] - true_step[:, 1]
        sq_horiz = dx**2 + dy**2
        horiz_rmse = float(np.sqrt(np.mean(sq_horiz)))

        # Vertical error in altitude (ft)
        dz = pred_step[:, 2] - true_step[:, 2]
        sq_vert = dz**2
        vert_rmse = float(np.sqrt(np.mean(sq_vert)))

        results.append(
            PositionErrorMetric(
                horizon_s=int(h),
                horizontal_rmse_nm=horiz_rmse,
                vertical_rmse_ft=vert_rmse,
                num_samples=n_samples,
            )
        )

    return results


def format_position_errors(metrics: Sequence[PositionErrorMetric]) -> str:
    """Format position error metrics into a readable string table."""
    lines: list[str] = [
        "----------------------------------------------------------------------",
        "Forecast Horizon (s) | Horizontal RMSE (NM) | Vertical RMSE (ft)",
        "----------------------------------------------------------------------",
    ]
    for m in metrics:
        lines.append(f"{m.horizon_s:>20} | {m.horizontal_rmse_nm:>20.4f} | {m.vertical_rmse_ft:>18.2f}")
    lines.append("----------------------------------------------------------------------")
    return "\n".join(lines)
