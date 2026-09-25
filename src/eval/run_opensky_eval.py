"""Evaluation of cv_smoothed baseline and frozen residual LSTM on OpenSky Network ADS-B data.

Research simulation only - not for operational use.
Evaluates trajectory position prediction error (Part 1 RMSE only).
Conflict evaluation is omitted because nominal operational ADS-B in controlled
airspace contains no verified ground-truth separation loss events.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.windows import extract_dataset_windows
from src.eval.metrics import evaluate_position_errors, format_position_errors
from src.models.baseline import SmoothedConstantVelocityPredictor
from src.models.lstm_predictor import LSTMTrajectoryPredictor


def evaluate_opensky(
    data_dir: str | Path = "data/opensky_live_sample",
    checkpoint: str | Path = "checkpoints/lstm_v1/best.pt",
) -> None:
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.parquet")))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    # Filter to flights with duration >= 360s (history 60s + horizon 300s)
    qualifying_files: list[Path] = []
    short_files_count = 0
    for f in all_files:
        try:
            df = pd.read_parquet(f, columns=["t"])
            if float(df["t"].max()) >= 360.0:
                qualifying_files.append(f)
            else:
                short_files_count += 1
        except Exception:
            pass

    print("=" * 70)
    print("OPENSKY REAL-DATA TRAJECTORY EVALUATION (PREDICTION ERROR ONLY)")
    print("=" * 70)
    print(f"Data Directory         : {data_path}")
    print(f"Total Live Flights     : {len(all_files)} ({short_files_count} discarded as < 6 min track)")
    print(f"Qualifying En-Route    : {len(qualifying_files)} flights (continuous track >= 6 min / 360s)")
    print(f"LSTM Checkpoint        : {checkpoint}")
    print(f"Window Specification   : 60s history (13 steps), 300s horizon (60 steps), dt=5s")
    print("=" * 70)

    if not qualifying_files:
        print("\nZero usable flights with duration >= 360s. Cannot extract windows. Stopping.")
        return

    # Window extraction
    histories, futures_true, metadata = extract_dataset_windows(
        scenario_files=qualifying_files,
        origin_step_s=30.0,
        history_s=60.0,
        horizon_s=300.0,
        dt_s=5.0,
        use_noisy_history=False,
    )
    print(f"\nTotal OpenSky live window samples extracted: {len(metadata)}")

    horizons = (30, 60, 120, 180, 240, 300)

    # 1. Evaluate cv_smoothed
    print("\n" + "=" * 70)
    print("EVALUATION: SMOOTHED CONSTANT-VELOCITY (cv_smoothed)")
    print("=" * 70)
    cv_predictor = SmoothedConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)
    cv_preds = cv_predictor.predict(histories)
    cv_metrics = evaluate_position_errors(cv_preds, futures_true, horizons_s=horizons, dt_s=5.0)
    print("[PART 1: TRAJECTORY POSITION PREDICTION ERROR - OPENSKY - cv_smoothed]")
    print(format_position_errors(cv_metrics))

    # 2. Evaluate residual LSTM
    print("\n" + "=" * 70)
    print(f"EVALUATION: RESIDUAL LSTM ({Path(checkpoint).name})")
    print("=" * 70)
    lstm_predictor = LSTMTrajectoryPredictor.from_checkpoint(checkpoint)
    lstm_preds = lstm_predictor.predict(histories)
    lstm_metrics = evaluate_position_errors(lstm_preds, futures_true, horizons_s=horizons, dt_s=5.0)
    print(f"[PART 1: TRAJECTORY POSITION PREDICTION ERROR - OPENSKY - lstm]")
    print(format_position_errors(lstm_metrics))

    # 3. Side-by-Side Comparison: OpenSky vs hard_large TEST
    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE RMSE COMPARISON: OPENSKY REAL ADS-B vs HARD_LARGE TEST")
    print("=" * 90)
    print(f"{'Horizon':>7} | {'cv_smoothed (OpenSky)':>21} | {'LSTM (OpenSky)':>21} | {'cv_sm (hard_large)':>20} | {'LSTM (hard_large)':>20}")
    print(f"{'(s)':>7} | {'Horiz(NM) / Vert(ft)':>21} | {'Horiz(NM) / Vert(ft)':>21} | {'Horiz(NM) / Vert(ft)':>20} | {'Horiz(NM) / Vert(ft)':>20}")
    print("-" * 90)

    # Reference values on hard_large TEST from benchmark logs
    hl_ref = {
        30: (0.1690, 65.72, 0.0979, 34.53),
        60: (0.3545, 108.15, 0.2526, 59.40),
        120: (0.8359, 189.22, 0.6907, 93.98),
        180: (1.4344, 267.48, 1.2555, 117.78),
        240: (2.1312, 344.94, 1.9233, 135.33),
        300: (2.9138, 421.58, 2.6798, 148.90),
    }

    for cv_m, lstm_m in zip(cv_metrics, lstm_metrics):
        h = cv_m.horizon_s
        hl_cv_h, hl_cv_v, hl_lstm_h, hl_lstm_v = hl_ref.get(h, (0, 0, 0, 0))
        op_cv_str = f"{cv_m.horizontal_rmse_nm:.4f} / {cv_m.vertical_rmse_ft:.2f}"
        op_lstm_str = f"{lstm_m.horizontal_rmse_nm:.4f} / {lstm_m.vertical_rmse_ft:.2f}"
        hl_cv_str = f"{hl_cv_h:.4f} / {hl_cv_v:.2f}"
        hl_lstm_str = f"{hl_lstm_h:.4f} / {hl_lstm_v:.2f}"
        print(f"{h:7d} | {op_cv_str:>21} | {op_lstm_str:>21} | {hl_cv_str:>20} | {hl_lstm_str:>20}")

    print("=" * 90)

    print("\n" + "=" * 70)
    print("NOTE ON CONFLICT EVALUATION:")
    print("Conflict evaluation is omitted on OpenSky data because nominal operational ADS-B in controlled airspace contains no verified ground-truth separation loss events.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory predictors on OpenSky ADS-B.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/opensky_live_sample",
        help="Path to OpenSky sample parquet directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lstm_v1/best.pt",
        help="Path to trained LSTM checkpoint.",
    )
    args = parser.parse_args()

    evaluate_opensky(data_dir=args.data_dir, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
