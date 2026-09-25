"""Path-consistency check for Residual LSTM trajectory predictor.

Research simulation only - not for operational use.
Compares LSTM prediction outputs under three inference paths on the first 2,000 windows of hard_large TEST:
(a) Single batch of 2,000 through forward path on device
(b) Chunks of 256 through forward path on device
(c) Origin-level loop as used inside conflict_eval (per-aircraft calls of shape 13x3)

Also verifies history column names read by extract_scenario_windows across hard_large and hard_v1,
and checks the columns stored in the training window cache.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data.splits import get_or_create_splits
from src.data.synthetic import TrajectoryConfig
from src.data.windows import extract_scenario_origin_groups, extract_scenario_windows
from src.models.lstm_predictor import LSTMTrajectoryPredictor


def check_history_columns() -> None:
    """Check which history columns are read on hard_large, hard_v1, and in training cache."""
    print("=" * 75)
    print("CHECK: HISTORY COLUMNS READ ACROSS DATASETS AND CACHE")
    print("=" * 75)

    # 1. hard_large
    hl_dir = Path("data/synthetic_hard_large")
    hl_sample = next(hl_dir.glob("*.parquet"))
    df_hl = pd.read_parquet(hl_sample)
    hl_cols = (
        ["x_obs_nm", "y_obs_nm", "alt_obs_ft"]
        if ("x_obs_nm" in df_hl.columns)
        else ["x_nm", "y_nm", "alt_ft"]
    )
    print(f"hard_large sample parquet columns : {list(df_hl.columns)}")
    print(f"hard_large extract_scenario_windows history columns : {hl_cols}")

    # 2. hard_v1
    h1_dir = Path("data/synthetic_hard")
    h1_sample = next(h1_dir.glob("*.parquet"))
    df_h1 = pd.read_parquet(h1_sample)
    h1_cols = (
        ["x_obs_nm", "y_obs_nm", "alt_obs_ft"]
        if ("x_obs_nm" in df_h1.columns)
        else ["x_nm", "y_nm", "alt_ft"]
    )
    print(f"hard_v1 sample parquet columns    : {list(df_h1.columns)}")
    print(f"hard_v1 extract_scenario_windows history columns    : {h1_cols}")

    # 3. Training window cache
    cache_path = Path("data/cache/hard_large_val_windows.pt")
    if cache_path.exists():
        cache_data = torch.load(cache_path, map_location="cpu")
        print(f"Training val cache path           : {cache_path}")
        print(f"Training val cache histories shape: {cache_data['histories'].shape}")
        print(f"Training val cache futures shape  : {cache_data['futures'].shape}")

        splits_file = Path("data/splits_hard_large.json")
        with open(splits_file, "r") as sf:
            val_splits = json.load(sf)
        first_val_id = val_splits["val"][0]
        df_first_val = pd.read_parquet(hl_dir / f"{first_val_id}.parquet")

        h_noisy, _, _ = extract_scenario_windows(df_first_val, use_noisy_history=True)
        h_clean, _, _ = extract_scenario_windows(df_first_val, use_noisy_history=False)
        first_cache = cache_data["histories"][0].numpy()
        diff_noisy = float(np.max(np.abs(first_cache - h_noisy[0])))
        diff_clean = float(np.max(np.abs(first_cache - h_clean[0])))
        is_noisy = diff_noisy < 1e-3
        print(f"Training val cache matches noisy columns: {is_noisy} (diff_noisy={diff_noisy:.6f}, diff_clean={diff_clean:.6f})")
    else:
        print(f"Training cache {cache_path} not found.")


def run_path_consistency_check(
    checkpoint_path: str = "checkpoints/lstm_v1/best.pt",
    config_path: str = "configs/default.yaml",
    target_windows: int = 2000,
) -> None:
    """Run 3-path consistency check on first 2,000 windows of hard_large TEST."""
    cfg = TrajectoryConfig.from_yaml(config_path, variant="hard_large")
    data_dir = Path("data/synthetic_hard_large")
    splits = get_or_create_splits(data_dir=data_dir, config_path=config_path, variant="hard_large", output_file=cfg.splits_file)
    test_ids = splits.get("test", [])

    print("\n" + "=" * 75)
    print(f"PATH CONSISTENCY CHECK ON {target_windows} WINDOWS (HARD_LARGE TEST SPLIT)")
    print("=" * 75)

    predictor = LSTMTrajectoryPredictor.from_checkpoint(checkpoint_path)
    device = predictor.device
    print(f"Model checkpoint : {checkpoint_path}")
    print(f"Active device    : {device}")

    # Collect exactly target_windows from test scenarios
    histories_list: list[np.ndarray] = []
    futures_true_list: list[np.ndarray] = []

    for sid in test_ids:
        f = data_dir / f"{sid}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        h, fut, meta = extract_scenario_windows(df, use_noisy_history=True)

        for i in range(len(h)):
            histories_list.append(h[i])
            futures_true_list.append(fut[i])
            if len(histories_list) == target_windows:
                break

        if len(histories_list) == target_windows:
            break

    histories_2k = np.array(histories_list[:target_windows], dtype=np.float32)  # (2000, 13, 3)
    futures_true_2k = np.array(futures_true_list[:target_windows], dtype=np.float32)  # (2000, 60, 3)

    print(f"Extracted windows : {len(histories_2k)} (shape: {histories_2k.shape})")

    # Path (a): Single batch of 2,000 through model forward path on device
    predictor.eval()
    with torch.no_grad():
        t_a = torch.as_tensor(histories_2k, dtype=torch.float32, device=device)
        pred_a_tensor = predictor.forward(t_a)
        assert isinstance(pred_a_tensor, torch.Tensor)
        pred_a = pred_a_tensor.detach().cpu().numpy()

    # Path (b): Chunks of 256 through forward path on device
    pred_b_chunks = []
    with torch.no_grad():
        for start in range(0, target_windows, 256):
            chunk = torch.as_tensor(histories_2k[start : start + 256], dtype=torch.float32, device=device)
            p_chunk = predictor.forward(chunk)
            assert isinstance(p_chunk, torch.Tensor)
            pred_b_chunks.append(p_chunk.detach().cpu().numpy())
    pred_b = np.concatenate(pred_b_chunks, axis=0)

    # Path (c): Origin-level loop as used inside conflict_eval
    # In conflict_eval: predictor.predict(group.history[acid]) for acid in aircraft_ids
    pred_c_list = []
    for h in histories_2k:
        p_c = predictor.predict(h)  # shape (60, 3)
        pred_c_list.append(p_c)
    pred_c = np.array(pred_c_list, dtype=np.float32)

    # Calculate differences between paths
    # Differences: (a) vs (b)
    diff_ab_horiz_nm = np.max(np.hypot(pred_a[..., 0] - pred_b[..., 0], pred_a[..., 1] - pred_b[..., 1]))
    diff_ab_vert_ft = np.max(np.abs(pred_a[..., 2] - pred_b[..., 2]))

    # Differences: (b) vs (c)
    diff_bc_horiz_nm = np.max(np.hypot(pred_b[..., 0] - pred_c[..., 0], pred_b[..., 1] - pred_c[..., 1]))
    diff_bc_vert_ft = np.max(np.abs(pred_b[..., 2] - pred_c[..., 2]))

    # Differences: (a) vs (c)
    diff_ac_horiz_nm = np.max(np.hypot(pred_a[..., 0] - pred_c[..., 0], pred_a[..., 1] - pred_c[..., 1]))
    diff_ac_vert_ft = np.max(np.abs(pred_a[..., 2] - pred_c[..., 2]))

    # RMSE @ 300 s under each path (Horizon 300 s is index 59)
    err_a_horiz = np.sqrt(np.mean((pred_a[:, 59, 0] - futures_true_2k[:, 59, 0])**2 + (pred_a[:, 59, 1] - futures_true_2k[:, 59, 1])**2))
    err_a_vert = np.sqrt(np.mean((pred_a[:, 59, 2] - futures_true_2k[:, 59, 2])**2))

    err_b_horiz = np.sqrt(np.mean((pred_b[:, 59, 0] - futures_true_2k[:, 59, 0])**2 + (pred_b[:, 59, 1] - futures_true_2k[:, 59, 1])**2))
    err_b_vert = np.sqrt(np.mean((pred_b[:, 59, 2] - futures_true_2k[:, 59, 2])**2))

    err_c_horiz = np.sqrt(np.mean((pred_c[:, 59, 0] - futures_true_2k[:, 59, 0])**2 + (pred_c[:, 59, 1] - futures_true_2k[:, 59, 1])**2))
    err_c_vert = np.sqrt(np.mean((pred_c[:, 59, 2] - futures_true_2k[:, 59, 2])**2))

    print("-" * 75)
    print("MAXIMUM ABSOLUTE DIFFERENCES BETWEEN INFERENCE PATHS:")
    print("-" * 75)
    print(f"Path (a) [single batch 2000] vs Path (b) [chunks of 256]:")
    print(f"  Horizontal max diff : {diff_ab_horiz_nm:.6f} NM")
    print(f"  Vertical max diff   : {diff_ab_vert_ft:.4f} ft")
    print()
    print(f"Path (b) [chunks of 256] vs Path (c) [conflict_eval loop]:")
    print(f"  Horizontal max diff : {diff_bc_horiz_nm:.6f} NM")
    print(f"  Vertical max diff   : {diff_bc_vert_ft:.4f} ft")
    print()
    print(f"Path (a) [single batch 2000] vs Path (c) [conflict_eval loop]:")
    print(f"  Horizontal max diff : {diff_ac_horiz_nm:.6f} NM")
    print(f"  Vertical max diff   : {diff_ac_vert_ft:.4f} ft")
    print()
    print("-" * 75)
    print("RMSE @ 300 s UNDER EACH PATH:")
    print("-" * 75)
    print(f"Path (a) single batch 2000  : Horizontal RMSE = {err_a_horiz:.4f} NM | Vertical RMSE = {err_a_vert:.2f} ft")
    print(f"Path (b) chunks of 256      : Horizontal RMSE = {err_b_horiz:.4f} NM | Vertical RMSE = {err_b_vert:.2f} ft")
    print(f"Path (c) conflict_eval loop : Horizontal RMSE = {err_c_horiz:.4f} NM | Vertical RMSE = {err_c_vert:.2f} ft")
    print("=" * 75)


def check_large_batch_mps_divergence(
    checkpoint_path: str = "checkpoints/lstm_v1/best.pt",
    cache_path: str = "data/cache/hard_large_val_windows.pt",
) -> None:
    """Demonstrate large-B unchunked MPS recurrent kernel divergence vs chunked and CPU."""
    c_p = Path(cache_path)
    if not c_p.exists():
        return

    print("\n" + "=" * 75)
    print("LARGE-BATCH MPS DIVERGENCE CHECK (SINGLE UNCHUNKED BATCH vs CHUNKED)")
    print("=" * 75)

    data = torch.load(c_p, map_location="cpu")
    hist_np = data["histories"].numpy()
    fut_np = data["futures"].numpy()
    b_total = len(hist_np)
    print(f"Testing on {b_total} real validation windows (B={b_total})...")

    model_mps = LSTMTrajectoryPredictor.from_checkpoint(checkpoint_path, device="mps")
    model_cpu = LSTMTrajectoryPredictor.from_checkpoint(checkpoint_path, device="cpu")

    with torch.no_grad():
        # 1. Single unchunked forward on MPS
        t_mps = torch.as_tensor(hist_np, dtype=torch.float32, device="mps")
        pred_mps_single = model_mps.forward(t_mps).cpu().numpy()

        # 2. Chunked (chunk_size=256) on MPS
        pred_mps_chunked = model_mps.predict(hist_np, chunk_size=256)

        # 3. CPU ground-truth reference
        t_cpu = torch.as_tensor(hist_np, dtype=torch.float32, device="cpu")
        pred_cpu_single = model_cpu.forward(t_cpu).numpy()

    # Max diffs
    diff_mps_single_vs_chunked_h = float(np.max(np.hypot(
        pred_mps_single[..., 0] - pred_mps_chunked[..., 0],
        pred_mps_single[..., 1] - pred_mps_chunked[..., 1],
    )))
    diff_mps_single_vs_chunked_v = float(np.max(np.abs(
        pred_mps_single[..., 2] - pred_mps_chunked[..., 2],
    )))

    diff_mps_chunked_vs_cpu_h = float(np.max(np.hypot(
        pred_mps_chunked[..., 0] - pred_cpu_single[..., 0],
        pred_mps_chunked[..., 1] - pred_cpu_single[..., 1],
    )))
    diff_mps_chunked_vs_cpu_v = float(np.max(np.abs(
        pred_mps_chunked[..., 2] - pred_cpu_single[..., 2],
    )))

    rmse_single_h = float(np.sqrt(np.mean((pred_mps_single[:, 59, 0] - fut_np[:, 59, 0])**2 + (pred_mps_single[:, 59, 1] - fut_np[:, 59, 1])**2)))
    rmse_single_v = float(np.sqrt(np.mean((pred_mps_single[:, 59, 2] - fut_np[:, 59, 2])**2)))

    rmse_chunked_h = float(np.sqrt(np.mean((pred_mps_chunked[:, 59, 0] - fut_np[:, 59, 0])**2 + (pred_mps_chunked[:, 59, 1] - fut_np[:, 59, 1])**2)))
    rmse_chunked_v = float(np.sqrt(np.mean((pred_mps_chunked[:, 59, 2] - fut_np[:, 59, 2])**2)))

    print(f"Single unchunked batch on MPS (B={b_total}) vs Chunked (256) on MPS:")
    print(f"  Horizontal max diff : {diff_mps_single_vs_chunked_h:.6f} NM")
    print(f"  Vertical max diff   : {diff_mps_single_vs_chunked_v:.4f} ft")
    print()
    print(f"Chunked (256) on MPS vs CPU Single Batch Reference:")
    print(f"  Horizontal max diff : {diff_mps_chunked_vs_cpu_h:.6f} NM")
    print(f"  Vertical max diff   : {diff_mps_chunked_vs_cpu_v:.4f} ft")
    print()
    print(f"Position RMSE @ 300 s under Single MPS Batch vs Chunked MPS:")
    print(f"  Single MPS batch (B={b_total}) : Horizontal = {rmse_single_h:.4f} NM | Vertical = {rmse_single_v:.2f} ft")
    print(f"  Chunked MPS (chunk=256)        : Horizontal = {rmse_chunked_h:.4f} NM | Vertical = {rmse_chunked_v:.2f} ft")
    print("=" * 75)


def main() -> None:
    check_history_columns()
    run_path_consistency_check()
    check_large_batch_mps_divergence()


if __name__ == "__main__":
    main()
