"""Live OpenSky ADS-B Collector and Track Reassembly.

Research simulation only - not for operational use.
Polls the OpenSky live REST API repeatedly over a bounding box,
prints each poll's timestamp and aircraft count live,
reassembles continuous tracks, and saves parquet files matching the schema.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import numpy as np
import pandas as pd


LAT_MIN, LAT_MAX = 49.0, 52.0
LON_MIN, LON_MAX = 7.0, 11.0
ALT_MIN_M = 3000.0  # FL100
LAT0, LON0 = 50.5, 9.0
API_URL = f"https://opensky-network.org/api/states/all?lamin={LAT_MIN}&lamax={LAT_MAX}&lomin={LON_MIN}&lomax={LON_MAX}"


def collect_live_tracks(
    duration_minutes: float = 20.0,
    poll_interval_s: float = 12.0,
    output_dir: Path = Path("data/opensky_live_sample"),
) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_duration_s = duration_minutes * 60.0
    start_time = time.time()
    end_time = start_time + total_duration_s

    print("=" * 70, flush=True)
    print("OPENSKY LIVE ADS-B STREAMING COLLECTOR", flush=True)
    print("=" * 70, flush=True)
    print(f"Target Duration        : {duration_minutes:.1f} minutes ({total_duration_s:.0f} seconds)", flush=True)
    print(f"Poll Interval          : {poll_interval_s:.1f} seconds", flush=True)
    print(f"Sector Bounding Box    : Lat [{LAT_MIN}°, {LAT_MAX}°] N, Lon [{LON_MIN}°, {LON_MAX}°] E", flush=True)
    print(f"Altitude Floor         : {ALT_MIN_M} m (~{ALT_MIN_M * 3.28084:.0f} ft)", flush=True)
    print(f"Live API Endpoint      : {API_URL}", flush=True)
    print(f"Output Directory       : {output_dir}", flush=True)
    print("=" * 70, flush=True)

    observations: list[dict] = []
    poll_idx = 0

    while time.time() < end_time:
        poll_idx += 1
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        req = urllib.request.Request(API_URL, headers={"User-Agent": "Mozilla/5.0"})

        total_in_bbox = 0
        en_route_count = 0

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                states = data.get("states", []) or []
                total_in_bbox = len(states)
                for s in states:
                    # OpenSky state vector fields:
                    # 0: icao24, 1: callsign, 2: origin_country, 3: time_position, 4: last_contact
                    # 5: longitude, 6: latitude, 7: baro_altitude, 8: on_ground, 9: velocity
                    # 10: true_track, 11: vertical_rate
                    if len(s) < 12:
                        continue
                    icao24 = s[0]
                    t_pos = s[3]
                    lon = s[5]
                    lat = s[6]
                    baro_alt = s[7]
                    on_ground = s[8]
                    vel = s[9]
                    trk = s[10]
                    v_rate = s[11]

                    if on_ground or baro_alt is None or lon is None or lat is None:
                        continue
                    if baro_alt < ALT_MIN_M:
                        continue

                    en_route_count += 1
                    t_obs = float(t_pos) if t_pos is not None else float(data.get("time", time.time()))
                    observations.append(
                        {
                            "icao24": str(icao24).strip(),
                            "t": t_obs,
                            "lat": float(lat),
                            "lon": float(lon),
                            "baroaltitude": float(baro_alt),
                            "velocity": float(vel) if vel is not None else 200.0,
                            "heading": float(trk) if trk is not None else 0.0,
                            "vertrate": float(v_rate) if v_rate is not None else 0.0,
                        }
                    )

            elapsed_s = time.time() - start_time
            print(
                f"[Poll {poll_idx:03d} | {now_utc} | Elapsed: {elapsed_s:5.1f}s] "
                f"BBox total: {total_in_bbox:2d} | En-route (>FL100): {en_route_count:2d} | "
                f"Accumulated obs: {len(observations)}",
                flush=True,
            )

        except urllib.error.HTTPError as e:
            print(f"[Poll {poll_idx:03d} | {now_utc}] HTTP Error {e.code}: {e.reason}", flush=True)
        except Exception as e:
            print(f"[Poll {poll_idx:03d} | {now_utc}] Request failed: {type(e).__name__}: {e}", flush=True)

        # Sleep remaining time until next poll
        time.sleep(poll_interval_s)

    print("\n" + "=" * 70, flush=True)
    print("LIVE POLLING COMPLETE - REASSEMBLING TRAJECTORIES", flush=True)
    print("=" * 70, flush=True)

    if not observations:
        print("Zero observations collected. Stopping.", flush=True)
        return 0, 0, 0

    df_obs = pd.DataFrame(observations)
    m_per_deg_lat = 60.0
    m_per_deg_lon = 60.0 * np.cos(np.radians(LAT0))

    valid_flights = 0
    total_points = 0
    usable_windows_count = 0
    metadata_records: list[dict] = []

    for icao24, group in df_obs.groupby("icao24"):
        group = group.sort_values("t").drop_duplicates(subset=["t"])
        t_raw = group["t"].values.astype(float)
        if len(t_raw) < 2:
            continue

        gaps = np.diff(t_raw)
        split_pts = [0] + (np.where(gaps > 35.0)[0] + 1).tolist() + [len(group)]
        segments = [group.iloc[split_pts[i] : split_pts[i + 1]] for i in range(len(split_pts) - 1)]

        for seg_idx, seg in enumerate(segments):
            dur = float(seg["t"].max() - seg["t"].min())
            if dur < 60.0 or len(seg) < 5:
                # Need at least 60s for history
                continue

            t_rel = (seg["t"] - seg["t"].min()).values.astype(float)
            t_grid = np.arange(0.0, float(t_rel.max()) + 1e-4, 5.0)

            x_raw = (seg["lon"].values.astype(float) - LON0) * m_per_deg_lon
            y_raw = (seg["lat"].values.astype(float) - LAT0) * m_per_deg_lat
            alt_raw = seg["baroaltitude"].values.astype(float) * 3.28084
            spd_raw = seg["velocity"].fillna(200.0).values.astype(float) * 1.94384
            hdg_rad = np.unwrap(np.radians(seg["heading"].fillna(0.0).values.astype(float)))

            x_interp = np.interp(t_grid, t_rel, x_raw)
            y_interp = np.interp(t_grid, t_rel, y_raw)
            alt_interp = np.interp(t_grid, t_rel, alt_raw)
            spd_interp = np.interp(t_grid, t_rel, spd_raw)
            hdg_interp = (np.degrees(np.interp(t_grid, t_rel, hdg_rad))) % 360.0

            flight_id = f"opensky_live_{icao24}" if len(segments) == 1 else f"opensky_live_{icao24}_{seg_idx}"
            acid = f"AC_{icao24}"

            flight_df = pd.DataFrame(
                {
                    "scenario_id": flight_id,
                    "aircraft_id": acid,
                    "t": t_grid,
                    "x_nm": x_interp,
                    "y_nm": y_interp,
                    "alt_ft": alt_interp,
                    "heading_deg": hdg_interp,
                    "speed_kt": spd_interp,
                }
            )

            out_file = output_dir / f"{flight_id}.parquet"
            flight_df.to_parquet(out_file, index=False)
            valid_flights += 1
            total_points += len(flight_df)

            # Check if long enough for 60s history + 300s horizon = 360s
            is_usable_window = dur >= 360.0
            if is_usable_window:
                # Number of origins = (dur - 360)/30 + 1
                n_win = int((dur - 360.0) // 30.0) + 1
                usable_windows_count += n_win

            metadata_records.append(
                {
                    "scenario_id": flight_id,
                    "aircraft_id": acid,
                    "icao24": str(icao24),
                    "duration_s": dur,
                    "n_points": len(flight_df),
                    "usable_for_300s_eval": is_usable_window,
                    "file": out_file.name,
                }
            )

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f_meta:
        json.dump(
            {
                "collection_start_utc": datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat(),
                "duration_minutes": duration_minutes,
                "poll_interval_s": poll_interval_s,
                "total_polls": poll_idx,
                "total_raw_observations": len(observations),
                "total_reassembled_tracks": valid_flights,
                "usable_windows_for_300s_eval": usable_windows_count,
                "tracks": metadata_records,
            },
            f_meta,
            indent=2,
        )

    print(f"Total Reassembled Real Flights : {valid_flights}", flush=True)
    print(f"Total Interpolated 5s Points   : {total_points}", flush=True)
    print(f"Usable 360s Windows (60h+300f) : {usable_windows_count}", flush=True)
    print(f"Parquet files saved to         : {output_dir}", flush=True)
    print("=" * 70, flush=True)

    return valid_flights, total_points, usable_windows_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live OpenSky state vectors and reassemble tracks.")
    parser.add_argument("--duration-minutes", type=float, default=20.0, help="Duration to poll in minutes.")
    parser.add_argument("--interval", type=float, default=12.0, help="Poll interval in seconds.")
    parser.add_argument("--output-dir", type=str, default="data/opensky_live_sample", help="Output directory.")
    args = parser.parse_args()

    collect_live_tracks(
        duration_minutes=args.duration_minutes,
        poll_interval_s=args.interval,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
