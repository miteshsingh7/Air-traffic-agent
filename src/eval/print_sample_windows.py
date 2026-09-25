"""Print 3 specific trajectory windows from hard_large TEST split.

Research simulation only - not for operational use.
Extracts:
1. Non-maneuvering aircraft window
2. Horizontal maneuver aircraft window
3. Vertical maneuver aircraft window

For each window, prints:
- 13x3 noisy history array
- cv_smoothed, lstm, and true-future positions at forecast horizons 60s, 180s, 300s.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.windows import extract_scenario_windows
from src.models.baseline import SmoothedConstantVelocityPredictor
from src.models.lstm_predictor import LSTMTrajectoryPredictor


def main() -> None:
    cv_pred = SmoothedConstantVelocityPredictor(dt_s=5.0, horizon_steps=60)
    lstm_pred = LSTMTrajectoryPredictor.from_checkpoint("checkpoints/lstm_v1/best.pt")

    samples = [
        ("NON-MANEUVERING", "scenario_0003", "AC_0", 60.0),
        ("HORIZONTAL MANEUVER", "scenario_0010", "AC_1", 240.0),
        ("VERTICAL MANEUVER", "scenario_0151", "AC_0", 360.0),
    ]

    horizon_indices = {60: 11, 180: 35, 300: 59}

    for label, sid, acid, origin_t in samples:
        print("=" * 80)
        print(f"SAMPLE WINDOW: {label} (scenario_id={sid}, aircraft_id={acid}, origin_t={origin_t}s)")
        print("=" * 80)
        df = pd.read_parquet(f"data/synthetic_hard_large/{sid}.parquet")
        h_arr, fut_arr, meta_arr = extract_scenario_windows(df, use_noisy_history=True)
        match_idx = [
            i
            for i, m in enumerate(meta_arr)
            if m.scenario_id == sid and m.aircraft_id == acid and abs(m.origin_t - origin_t) < 1e-3
        ][0]

        h = h_arr[match_idx]  # (13, 3)
        f_true = fut_arr[match_idx]  # (60, 3)

        f_cv = cv_pred.predict(h)  # (60, 3)
        f_lstm = lstm_pred.predict(h)  # (60, 3)

        print("13x3 History Array [x_obs_nm, y_obs_nm, alt_obs_ft] (t = -60s to 0s relative to origin):")
        print("Step | t_rel (s) | x_obs (NM) | y_obs (NM) | alt_obs (ft)")
        print("-" * 55)
        for step in range(13):
            t_rel = -60.0 + step * 5.0
            print(f"{step:4d} | {t_rel:9.1f} | {h[step, 0]:10.4f} | {h[step, 1]:10.4f} | {h[step, 2]:12.2f}")
        print("-" * 55)

        print("\nPredicted vs True Positions at Forecast Horizons 60s, 180s, 300s:")
        print(f"Horizon | Method      | x (NM)     | y (NM)     | alt (ft)   | Horiz Err (NM) | Vert Err (ft)")
        print("-" * 80)
        for h_s in [60, 180, 300]:
            idx = horizon_indices[h_s]
            p_true = f_true[idx]
            p_cv = f_cv[idx]
            p_lstm = f_lstm[idx]

            cv_h_err = float(np.hypot(p_cv[0] - p_true[0], p_cv[1] - p_true[1]))
            cv_v_err = float(abs(p_cv[2] - p_true[2]))
            lstm_h_err = float(np.hypot(p_lstm[0] - p_true[0], p_lstm[1] - p_true[1]))
            lstm_v_err = float(abs(p_lstm[2] - p_true[2]))

            print(f"{h_s:5d}s  | TRUE        | {p_true[0]:10.4f} | {p_true[1]:10.4f} | {p_true[2]:10.2f} |              - |             -")
            print(f"       | cv_smoothed | {p_cv[0]:10.4f} | {p_cv[1]:10.4f} | {p_cv[2]:10.2f} | {cv_h_err:14.4f} | {cv_v_err:13.2f}")
            print(f"       | lstm        | {p_lstm[0]:10.4f} | {p_lstm[1]:10.4f} | {p_lstm[2]:10.2f} | {lstm_h_err:14.4f} | {lstm_v_err:13.2f}")
            print("-" * 80)
        print()


if __name__ == "__main__":
    main()
