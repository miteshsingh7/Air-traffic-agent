"""Training pipeline for per-aircraft residual LSTM trajectory predictor.

Research simulation only - not for operational use.
Trains exclusively on the hard_large TRAIN split. Validation split is used
solely for checkpoint selection and early stopping. Test split is never touched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import get_or_create_splits
from src.data.synthetic import TrajectoryConfig
from src.data.windows import extract_scenario_windows
from src.models.lstm_predictor import LSTMTrajectoryPredictor


def load_split_windows(
    data_dir: Path,
    scenario_ids: Sequence[str],
    cache_path: Path | None = None,
    max_workers: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract or load cached history and future tensors for given scenarios."""
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached windows from {cache_path}...")
        data = torch.load(cache_path)
        return data["histories"], data["futures"]

    files = [data_dir / f"{sid}.parquet" for sid in scenario_ids if (data_dir / f"{sid}.parquet").exists()]
    if not files:
        raise FileNotFoundError(f"No scenario files found in {data_dir} for {len(scenario_ids)} IDs")

    def _extract_one(f: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_parquet(f)
        h, fut, _ = extract_scenario_windows(df, use_noisy_history=True)
        return h, fut

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_extract_one, files))

    all_h = [r[0] for r in results if len(r[0]) > 0]
    all_f = [r[1] for r in results if len(r[1]) > 0]

    h_arr = np.concatenate(all_h, axis=0).astype(np.float32)
    f_arr = np.concatenate(all_f, axis=0).astype(np.float32)

    h_tensor = torch.from_numpy(h_arr)
    f_tensor = torch.from_numpy(f_arr)

    dt = time.time() - t0
    print(f"Extracted {len(h_tensor)} windows from {len(files)} scenarios in {dt:.1f}s")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"histories": h_tensor, "futures": f_tensor}, cache_path)
        print(f"Saved window cache to {cache_path}")

    return h_tensor, f_tensor


def train(
    data_dir: str | Path = "data/synthetic_hard_large",
    config_path: str | Path = "configs/default.yaml",
    splits_file: str | Path = "data/splits_hard_large.json",
    checkpoint_dir: str | Path = "checkpoints/lstm_v1",
    batch_size: int = 256,
    max_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
    device: str | None = None,
) -> Path:
    """Train the residual LSTM model on hard_large TRAIN split."""
    # 1. Device selection and reproducible seeds
    if device is None:
        if torch.backends.mps.is_available():
            target_device = torch.device("mps")
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 75)
    print("RESIDUAL LSTM TRAJECTORY PREDICTOR TRAINING PIPELINE")
    print("=" * 75)
    print(f"Target Device: {target_device}")
    print(f"Random Seed  : {seed}")

    # 2. Strict dataset split isolation verification
    d_dir = Path(data_dir)
    cfg = TrajectoryConfig.from_yaml(config_path, variant="hard_large")
    splits = get_or_create_splits(
        data_dir=d_dir,
        config_path=config_path,
        variant="hard_large",
        output_file=splits_file,
    )

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # Strict assertions that test scenarios are NEVER included
    assert train_ids.isdisjoint(test_ids), "CRITICAL: Train split contains test scenarios!"
    assert val_ids.isdisjoint(test_ids), "CRITICAL: Val split contains test scenarios!"
    assert train_ids.isdisjoint(val_ids), "CRITICAL: Train and val splits overlap!"

    print(f"Splits verified: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test scenarios.")
    print("Test split is strictly isolated and excluded from training and checkpoint selection.")

    # 3. Load or cache windows
    cache_dir = Path("data/cache")
    train_cache = cache_dir / "hard_large_train_windows.pt"
    val_cache = cache_dir / "hard_large_val_windows.pt"

    train_h, train_f = load_split_windows(d_dir, splits["train"], cache_path=train_cache)
    val_h, val_f = load_split_windows(d_dir, splits["val"], cache_path=val_cache)

    train_dataset = TensorDataset(train_h, train_f)
    val_dataset = TensorDataset(val_h, val_f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # 4. Instantiate model
    model = LSTMTrajectoryPredictor(
        dt_s=cfg.resample_rate_s,
        horizon_steps=int(round(cfg.horizon_s / cfg.resample_rate_s)),
        history_steps=int(round(cfg.history_s / cfg.resample_rate_s)) + 1,
        hidden_size=128,
        num_layers=2,
        device=target_device,
    )
    n_params = model.count_parameters()
    print(f"Model Architecture: 2-layer LSTM (hidden=128) + Residual MLP Head")
    print(f"Total Trainable Parameters: {n_params:,}")

    # 5. Optimizer, scheduler, criterion
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    best_val_rmse = float("inf")
    patience_counter = 0

    print("-" * 75)
    print(f"{'Epoch':<8} | {'TrainLoss':<9} | {'ValLoss':<9} | {'Val Horiz RMSE (60/180/300s)':<30} | {'Val Vert 300s':<12} | {'Time':<6}")
    print("-" * 75)

    # Time-step indices for horizon metrics (5s step): 60s=step 12 (idx 11), 180s=step 36 (idx 35), 300s=step 60 (idx 59)
    step_60 = 11
    step_180 = 35
    step_300 = 59

    scale_dev = model.scale.to(target_device)

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.time()

        # Training phase
        model.train()
        train_loss_total = 0.0

        for b_hist, b_fut in train_loader:
            b_hist = b_hist.to(target_device)
            b_fut = b_fut.to(target_device)

            optimizer.zero_grad()
            _, delta_norm = model(b_hist, return_residual=True)
            baseline_cv = model.compute_baseline_cv(b_hist)
            target_residual_norm = (b_fut - baseline_cv) / scale_dev

            loss = criterion(delta_norm, target_residual_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_total += loss.item() * len(b_hist)

        train_loss = train_loss_total / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0

        sq_err_h_60 = 0.0
        sq_err_h_180 = 0.0
        sq_err_h_300 = 0.0
        sq_err_v_300 = 0.0

        with torch.no_grad():
            for b_hist, b_fut in val_loader:
                b_hist = b_hist.to(target_device)
                b_fut = b_fut.to(target_device)

                y_pred, delta_norm = model(b_hist, return_residual=True)
                baseline_cv = model.compute_baseline_cv(b_hist)
                target_residual_norm = (b_fut - baseline_cv) / scale_dev
                v_loss = criterion(delta_norm, target_residual_norm)
                val_loss_total += v_loss.item() * len(b_hist)

                # Error metrics
                diff = y_pred - b_fut  # (B, 60, 3) [x, y, alt]
                h_err_sq = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2  # (B, 60)
                v_err_sq = diff[:, :, 2] ** 2  # (B, 60)

                sq_err_h_60 += float(torch.sum(h_err_sq[:, step_60]).item())
                sq_err_h_180 += float(torch.sum(h_err_sq[:, step_180]).item())
                sq_err_h_300 += float(torch.sum(h_err_sq[:, step_300]).item())
                sq_err_v_300 += float(torch.sum(v_err_sq[:, step_300]).item())
                val_samples += len(b_hist)

        val_loss = val_loss_total / val_samples
        rmse_h_60 = np.sqrt(sq_err_h_60 / val_samples)
        rmse_h_180 = np.sqrt(sq_err_h_180 / val_samples)
        rmse_h_300 = np.sqrt(sq_err_h_300 / val_samples)
        rmse_v_300 = np.sqrt(sq_err_v_300 / val_samples)

        # Primary selection metric: total position RMSE at 300s horizon
        val_metric = rmse_h_300

        scheduler.step(val_loss)
        epoch_dur = time.time() - t_epoch_start

        # Checkpoint selection
        is_best = val_metric < best_val_rmse
        best_str = ""
        if is_best:
            best_val_rmse = val_metric
            patience_counter = 0
            model.save_checkpoint(best_path)
            best_str = " * BEST"
        else:
            patience_counter += 1

        model.save_checkpoint(last_path)

        rmse_str = f"[{rmse_h_60:.4f}, {rmse_h_180:.4f}, {rmse_h_300:.4f}] NM"
        v_str = f"{rmse_v_300:.1f} ft"
        print(
            f"{epoch:02d}/{max_epochs:02d}    | "
            f"{train_loss:.5f}   | "
            f"{val_loss:.5f}   | "
            f"{rmse_str:<30} | "
            f"{v_str:<12} | "
            f"{epoch_dur:4.1f}s{best_str}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience {patience} exceeded).")
            break

    print("-" * 75)
    print(f"Training Complete. Best Validation RMSE @ 300s: {best_val_rmse:.4f} NM")
    print(f"Best checkpoint saved to: {best_path}")
    print(f"Last checkpoint saved to: {last_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual LSTM trajectory predictor.")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_hard_large", help="Dataset directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration YAML")
    parser.add_argument("--splits-file", type=str, default="data/splits_hard_large.json", help="Splits JSON")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lstm_v1", help="Checkpoint output dir")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Target device (mps, cuda, cpu)")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        config_path=args.config,
        splits_file=args.splits_file,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
