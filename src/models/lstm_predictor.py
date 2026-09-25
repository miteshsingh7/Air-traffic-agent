"""Per-aircraft residual LSTM trajectory predictor.

Research simulation only - not for operational use.
Estimates nonlinear trajectory residuals on top of smoothed constant-velocity baseline.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class LSTMTrajectoryPredictor(nn.Module):
    """Residual LSTM trajectory predictor combining physics baseline with learned corrections.

    Input:
        13-step noisy history [x_obs_nm, y_obs_nm, alt_obs_ft].
    Features (dim=6 per step):
        - Relative position: (p_t - p_12) / scale (scale: 10 NM lateral, 1000 ft vertical)
        - Step delta: (p_t - p_{t-1}) / scale (with delta_0 duplicated from delta_1)
    Backbone:
        - 2-layer LSTM (hidden_size=128)
    Head:
        - MLP mapping final hidden state to 60 x 3 = 180 normalized residual outputs.
        - Final layer initialized to 0 (weights & bias), ensuring epoch 0 identically matches cv_smoothed.
    Output:
        y_pred = y_cv_smoothed + delta_pred
    """

    def __init__(
        self,
        dt_s: float = 5.0,
        horizon_steps: int = 60,
        history_steps: int = 13,
        hidden_size: int = 128,
        num_layers: int = 2,
        lateral_scale_nm: float = 10.0,
        vertical_scale_ft: float = 1000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dt_s = dt_s
        self.horizon_steps = horizon_steps
        self.history_steps = history_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lateral_scale_nm = lateral_scale_nm
        self.vertical_scale_ft = vertical_scale_ft

        if device is None:
            if torch.backends.mps.is_available():
                self.target_device = torch.device("mps")
            elif torch.cuda.is_available():
                self.target_device = torch.device("cuda")
            else:
                self.target_device = torch.device("cpu")
        else:
            self.target_device = torch.device(device)

        # Scale buffer for lateral (x, y) and vertical (alt)
        scale_vec = torch.tensor(
            [lateral_scale_nm, lateral_scale_nm, vertical_scale_ft],
            dtype=torch.float32,
        )
        self.register_buffer("scale", scale_vec)

        # Centered time indices and denominator for vectorized least-squares baseline fit
        k_indices = torch.arange(history_steps, dtype=torch.float32)
        k_mean = (history_steps - 1.0) / 2.0
        k_centered = k_indices - k_mean
        denom = float(torch.sum(k_centered**2))
        self.register_buffer("k_centered", k_centered)
        self.register_buffer("k_denom", torch.tensor(denom, dtype=torch.float32))
        self.register_buffer(
            "current_k_offset",
            torch.tensor((history_steps - 1.0) - k_mean, dtype=torch.float32),
        )

        # Future time offsets for baseline extrapolation: [1, 2, ..., horizon_steps]
        h_offsets = torch.arange(1, horizon_steps + 1, dtype=torch.float32)
        self.register_buffer("h_offsets", h_offsets)

        # 2-layer LSTM backbone: input_size=6, hidden_size=128
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Residual MLP head: hidden_size -> 128 -> ReLU -> 180
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon_steps * 3),
        )

        # Zero-initialize the final linear layer so residual output is initially 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.to(self.target_device)

    @property
    def device(self) -> torch.device:
        return self.target_device

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, history: torch.Tensor) -> torch.Tensor:
        """Extract 6D normalized trajectory features from history.

        Parameters
        ----------
        history : torch.Tensor
            Batch history of shape (B, K, 3).

        Returns
        -------
        torch.Tensor
            Normalized features of shape (B, K, 6).
        """
        # Relative position: (p_t - p_{-1}) / scale
        origin_pos = history[:, -1:, :]  # (B, 1, 3)
        rel_pos = (history - origin_pos) / self.scale  # (B, K, 3)

        # Step deltas: (p_t - p_{t-1}) / scale
        deltas = (history[:, 1:, :] - history[:, :-1, :]) / self.scale  # (B, K-1, 3)
        delta_0 = deltas[:, :1, :]  # Duplicate delta_1 for step 0
        step_deltas = torch.cat([delta_0, deltas], dim=1)  # (B, K, 3)

        # Combined features: (B, K, 6)
        features = torch.cat([rel_pos, step_deltas], dim=2)
        return features

    def compute_baseline_cv(self, history: torch.Tensor) -> torch.Tensor:
        """Vectorized least-squares constant-velocity extrapolation matching cv_smoothed.

        Parameters
        ----------
        history : torch.Tensor
            Batch history of shape (B, K, 3).

        Returns
        -------
        torch.Tensor
            Predicted baseline futures of shape (B, H, 3).
        """
        y_mean = torch.mean(history, dim=1)
        slope = torch.sum(history * self.k_centered.view(1, -1, 1), dim=1) / self.k_denom
        pos_current_fit = y_mean + slope * self.current_k_offset
        future_base = (
            pos_current_fit.unsqueeze(1) + self.h_offsets.view(1, -1, 1) * slope.unsqueeze(1)
        )
        return future_base

    def forward(
        self,
        history: torch.Tensor,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting future positions and optional normalized residual.

        Parameters
        ----------
        history : torch.Tensor
            History positions of shape (B, K, 3).
        return_residual : bool, default False
            If True, also returns normalized residual of shape (B, H, 3).

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Full trajectory prediction (B, H, 3), and optionally normalized residual (B, H, 3).
        """
        b_size = history.shape[0]
        if history.device != self.scale.device:
            history = history.to(self.scale.device)

        # 1. Feature extraction
        features = self.extract_features(history)  # (B, K, 6)

        # 2. LSTM encoding
        lstm_out, _ = self.lstm(features)  # (B, K, hidden_size)
        final_state = lstm_out[:, -1, :]  # (B, hidden_size)

        # 3. Head prediction (normalized residual)
        delta_norm = self.head(final_state).view(b_size, self.horizon_steps, 3)  # (B, H, 3)

        # 4. Unscale residual to physical coordinates
        delta_physical = delta_norm * self.scale  # (B, H, 3)

        # 5. Physics baseline
        baseline_pred = self.compute_baseline_cv(history)  # (B, H, 3)

        # 6. Combined trajectory prediction
        y_pred = baseline_pred + delta_physical

        if return_residual:
            return y_pred, delta_norm
        return y_pred

    def predict(self, history: np.ndarray, chunk_size: int = 256) -> np.ndarray:
        """Predict future trajectory adhering to TrajectoryPredictor protocol.

        Parameters
        ----------
        history : np.ndarray
            History positions of shape (13, 3) or (B, 13, 3).
        chunk_size : int, default 256
            Maximum batch size processed per forward pass to avoid hardware/MPS
            memory issues and large-batch recurrent kernel degradation.

        Returns
        -------
        np.ndarray
            Predicted future positions of shape (60, 3) or (B, 60, 3).
        """
        self.eval()
        is_2d = history.ndim == 2
        if is_2d:
            h_in = history[np.newaxis, :, :]
        elif history.ndim == 3:
            h_in = history
        else:
            raise ValueError(f"Expected history ndim 2 or 3, got shape {history.shape}")

        b_total = h_in.shape[0]
        preds: list[np.ndarray] = []

        with torch.no_grad():
            for start_idx in range(0, b_total, chunk_size):
                chunk = h_in[start_idx : start_idx + chunk_size]
                t_in = torch.as_tensor(chunk, dtype=torch.float32, device=self.target_device)
                pred_t = self.forward(t_in, return_residual=False)
                assert isinstance(pred_t, torch.Tensor)
                preds.append(pred_t.detach().cpu().numpy())

        pred_np = np.concatenate(preds, axis=0)
        return pred_np[0] if is_2d else pred_np

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and architectural hyperparameters."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "dt_s": self.dt_s,
                "horizon_steps": self.horizon_steps,
                "history_steps": self.history_steps,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "lateral_scale_nm": self.lateral_scale_nm,
                "vertical_scale_ft": self.vertical_scale_ft,
            },
            p,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> LSTMTrajectoryPredictor:
        """Load trained LSTMTrajectoryPredictor from a checkpoint."""
        if device is None:
            if torch.backends.mps.is_available():
                target_device = torch.device("mps")
            elif torch.cuda.is_available():
                target_device = torch.device("cuda")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(device)

        checkpoint = torch.load(path, map_location=target_device)
        model = cls(
            dt_s=checkpoint.get("dt_s", 5.0),
            horizon_steps=checkpoint.get("horizon_steps", 60),
            history_steps=checkpoint.get("history_steps", 13),
            hidden_size=checkpoint.get("hidden_size", 128),
            num_layers=checkpoint.get("num_layers", 2),
            lateral_scale_nm=checkpoint.get("lateral_scale_nm", 10.0),
            vertical_scale_ft=checkpoint.get("vertical_scale_ft", 1000.0),
            device=target_device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(target_device)
        model.eval()
        return model
