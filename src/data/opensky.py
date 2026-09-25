"""OpenSky Network historical ADS-B data ingestion and preprocessing.

Research simulation only - not for operational use.

Sector Specification:
- Sector Name: Central European En-Route Sector (Frankfurt / Maastricht UAC FIR)
- Latitude Bounds: [49.0°, 52.0°] N
- Longitude Bounds: [7.0°, 11.0°] E
- Altitude Floor: 3,000 m (~FL100 / 9,842.5 ft) to isolate en-route traffic
- Reference Date: 2017-06-05 (00:00:00 to 01:00:00 UTC)
- Projection Origin: (lat0=50.5° N, lon0=9.0° E)
  x_nm = (lon - 9.0) * 60.0 * cos(50.5 * pi / 180)
  y_nm = (lat - 50.5) * 60.0
  alt_ft = baroaltitude_m * 3.28084
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile
import urllib.error
import urllib.request
import numpy as np
import pandas as pd


SECTOR_NAME = "Central European En-Route Sector (Frankfurt / Maastricht FIR)"
LAT_MIN, LAT_MAX = 49.0, 52.0
LON_MIN, LON_MAX = 7.0, 11.0
ALT_MIN_M = 3000.0  # ~9,842 ft (FL100)
LAT0, LON0 = 50.5, 9.0
HISTORICAL_DATE = "2017-06-05"
HISTORICAL_HOUR = "00"
S3_URL = (
    f"https://s3.opensky-network.org/data-samples/states/.{HISTORICAL_DATE}/"
    f"{HISTORICAL_HOUR}/states_{HISTORICAL_DATE}-{HISTORICAL_HOUR}.csv.tar"
)
REST_HISTORICAL_URL = "https://opensky-network.org/api/states/all?time=1496620800"
REST_LIVE_URL = "https://opensky-network.org/api/states/all"


def probe_opensky_rest_api() -> dict[str, str]:
    """Probe OpenSky REST endpoints to verify reachability and permissions."""
    results: dict[str, str] = {}

    # 1. Historical REST endpoint
    try:
        req = urllib.request.Request(REST_HISTORICAL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            results["rest_historical"] = f"HTTP {resp.status} OK"
    except urllib.error.HTTPError as e:
        results["rest_historical"] = f"HTTP {e.code} {e.reason} (anonymous historical REST query restricted)"
    except Exception as e:
        results["rest_historical"] = f"Failed: {e}"

    # 2. Live REST endpoint
    try:
        req = urllib.request.Request(REST_LIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            ac_count = len(data.get("states", []))
            results["rest_live"] = f"HTTP {resp.status} OK ({ac_count} live aircraft snapshots)"
    except Exception as e:
        results["rest_live"] = f"Failed: {e}"

    return results


def download_or_get_archive(cache_path: Path) -> Path:
    """Ensure the historical bulk tar archive is available locally."""
    if cache_path.exists() and cache_path.stat().st_size > 10_000_000:
        return cache_path

    print(f"Downloading OpenSky bulk sample from {S3_URL} -> {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(S3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(cache_path, "wb") as f_out:
        chunk_size = 1024 * 1024
        while chunk := resp.read(chunk_size):
            f_out.write(chunk)
    print(f"Downloaded {cache_path.stat().st_size / (1024 * 1024):.1f} MB.")
    return cache_path


def process_opensky_sector_data(
    tar_path: Path,
    output_dir: Path,
    min_continuous_duration_s: float = 360.0,
    max_gap_s: float = 25.0,
    dt_s: float = 5.0,
) -> tuple[int, int, list[dict]]:
    """Extract, filter, resample, and save sector flight trajectories as parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_gz_name = f"states_{HISTORICAL_DATE}-{HISTORICAL_HOUR}.csv.gz"
    print(f"Extracting sector records from {tar_path} ({csv_gz_name})...")

    with tarfile.open(tar_path, "r") as tar:
        member = tar.extractfile(csv_gz_name)
        if member is None:
            raise FileNotFoundError(f"{csv_gz_name} not found in archive {tar_path}")

        chunks: list[pd.DataFrame] = []
        use_cols = [
            "time",
            "icao24",
            "lat",
            "lon",
            "baroaltitude",
            "velocity",
            "heading",
            "onground",
        ]
        for chunk in pd.read_csv(member, compression="gzip", chunksize=1_000_000, usecols=use_cols):
            mask = (
                (chunk["lat"] >= LAT_MIN)
                & (chunk["lat"] <= LAT_MAX)
                & (chunk["lon"] >= LON_MIN)
                & (chunk["lon"] <= LON_MAX)
                & (chunk["baroaltitude"] > ALT_MIN_M)
                & (~chunk["onground"])
            )
            sub = chunk[mask].dropna(subset=["lat", "lon", "baroaltitude"])
            if len(sub) > 0:
                chunks.append(sub)

    if not chunks:
        raise ValueError("No sector records found matching filter criteria.")

    df_sector = pd.concat(chunks, ignore_index=True)
    print(f"Total raw sector observations: {len(df_sector)}")

    flights_processed = 0
    total_samples = 0
    metadata_records: list[dict] = []

    m_per_deg_lat = 60.0
    m_per_deg_lon = 60.0 * np.cos(np.radians(LAT0))

    for icao24, group in df_sector.groupby("icao24"):
        group = group.sort_values("time").drop_duplicates(subset=["time"])
        t_raw = group["time"].values.astype(float)
        if len(t_raw) < 2:
            continue

        gaps = np.diff(t_raw)
        split_pts = [0] + (np.where(gaps > max_gap_s)[0] + 1).tolist() + [len(group)]
        segments = [group.iloc[split_pts[i]:split_pts[i+1]] for i in range(len(split_pts) - 1)]

        for seg_idx, seg in enumerate(segments):
            dur = float(seg["time"].max() - seg["time"].min())
            if dur < min_continuous_duration_s or len(seg) < 20:
                continue

            t_rel = (seg["time"] - seg["time"].min()).values.astype(float)
            t_grid = np.arange(0.0, float(t_rel.max()) + 1e-4, dt_s)

            # Flat-earth equidistant projection
            x_raw = (seg["lon"].values.astype(float) - LON0) * m_per_deg_lon
            y_raw = (seg["lat"].values.astype(float) - LAT0) * m_per_deg_lat
            alt_raw = seg["baroaltitude"].values.astype(float) * 3.28084
            spd_raw = seg["velocity"].fillna(200.0).values.astype(float) * 1.94384
            hdg_rad = np.unwrap(np.radians(seg["heading"].fillna(0.0).values.astype(float)))

            # Linear interpolation to uniform 5s grid
            x_interp = np.interp(t_grid, t_rel, x_raw)
            y_interp = np.interp(t_grid, t_rel, y_raw)
            alt_interp = np.interp(t_grid, t_rel, alt_raw)
            spd_interp = np.interp(t_grid, t_rel, spd_raw)
            hdg_interp = (np.degrees(np.interp(t_grid, t_rel, hdg_rad))) % 360.0

            flight_id = f"opensky_{icao24}" if len(segments) == 1 else f"opensky_{icao24}_{seg_idx}"
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
            flights_processed += 1
            total_samples += len(flight_df)

            metadata_records.append(
                {
                    "scenario_id": flight_id,
                    "aircraft_id": acid,
                    "icao24": str(icao24),
                    "duration_s": dur,
                    "n_points": len(flight_df),
                    "file": out_file.name,
                }
            )

    # Save metadata summary
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f_meta:
        json.dump(
            {
                "sector": SECTOR_NAME,
                "lat_bounds": [LAT_MIN, LAT_MAX],
                "lon_bounds": [LON_MIN, LON_MAX],
                "alt_min_m": ALT_MIN_M,
                "projection_center": [LAT0, LON0],
                "date": HISTORICAL_DATE,
                "hour_utc": HISTORICAL_HOUR,
                "total_flights": flights_processed,
                "total_points": total_samples,
                "flights": metadata_records,
            },
            f_meta,
            indent=2,
        )

    return flights_processed, total_samples, metadata_records


def main() -> None:
    """CLI entrypoint for pulling and processing OpenSky sector trajectories."""
    parser = argparse.ArgumentParser(description="Ingest and preprocess OpenSky historical ADS-B data.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/opensky_sample",
        help="Directory to save extracted parquet files.",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default="/tmp/test_download.tar",
        help="Local path to OpenSky tar archive.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    archive_path = Path(args.archive_path)

    print("=" * 70)
    print("OPENSKY NETWORK HISTORICAL ADS-B INGESTION PIPELINE")
    print("=" * 70)
    print(f"Sector                 : {SECTOR_NAME}")
    print(f"Bounds                 : Lat [{LAT_MIN}°, {LAT_MAX}°] N, Lon [{LON_MIN}°, {LON_MAX}°] E")
    print(f"Altitude floor         : {ALT_MIN_M} m (~{ALT_MIN_M * 3.28084:.0f} ft / FL100)")
    print(f"Date & Hour (UTC)      : {HISTORICAL_DATE} {HISTORICAL_HOUR}:00:00")
    print(f"Projection Center      : ({LAT0}° N, {LON0}° E)")
    print("-" * 70)

    print("Probing OpenSky Network API endpoints:")
    api_probe = probe_opensky_rest_api()
    for endpoint, status in api_probe.items():
        print(f"  - {endpoint}: {status}")
    print("-" * 70)

    # Acquire archive
    tar_file = download_or_get_archive(archive_path)

    # Process sector
    n_flights, n_pts, meta = process_opensky_sector_data(
        tar_path=tar_file,
        output_dir=out_dir,
        min_continuous_duration_s=360.0,
        max_gap_s=25.0,
        dt_s=5.0,
    )

    print(f"\nProcessing Complete:")
    print(f"  - Filtered En-Route Flights (>= 6 min): {n_flights}")
    print(f"  - Resampled 5s Track Points Total    : {n_pts}")
    print(f"  - Output Parquet Directory           : {out_dir}")
    print(f"  - Metadata JSON                      : {out_dir / 'metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
