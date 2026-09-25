"""Dataset integrity and hashing utilities.

Research simulation only - not for operational use.
Computes and verifies SHA-256 over sorted parquet filenames and byte sizes,
combined with simulation configuration parameters and random seed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data.synthetic import TrajectoryConfig


def compute_parquet_sha256(data_dir: str | Path) -> str:
    """Compute SHA-256 digest over sorted parquet filenames and file sizes."""
    p_dir = Path(data_dir)
    parquet_files = sorted(p_dir.glob("*.parquet"), key=lambda p: p.name)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir} to hash")

    hasher = hashlib.sha256()
    for pf in parquet_files:
        line = f"{pf.name}:{pf.stat().st_size}\n"
        hasher.update(line.encode("utf-8"))

    return hasher.hexdigest()


def compute_parquet_content_sha256(data_dir: str | Path) -> str:
    """Compute SHA-256 digest over the raw bytes of every parquet file, sorted by filename."""
    p_dir = Path(data_dir)
    parquet_files = sorted(p_dir.glob("*.parquet"), key=lambda p: p.name)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir} to hash")

    hasher = hashlib.sha256()
    for pf in parquet_files:
        with open(pf, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

    return hasher.hexdigest()


def create_dataset_hash_file(
    data_dir: str | Path,
    config: TrajectoryConfig,
    dataset_name: str,
    scenario_count: int | None = None,
    hash_filename: str = "DATASET_HASH.txt",
) -> Path:
    """Generate and write a dataset integrity hash file with both metadata and content hashes."""
    p_dir = Path(data_dir)
    parquet_files = sorted(p_dir.glob("*.parquet"), key=lambda p: p.name)
    count = scenario_count if scenario_count is not None else len(parquet_files)
    sha256_hash = compute_parquet_sha256(p_dir)
    content_hash = compute_parquet_content_sha256(p_dir)

    lines = [
        f"dataset_name: {dataset_name}",
        f"dataset_variant: {config.dataset_variant}",
        f"seed: {config.seed}",
        f"scenario_count: {count}",
        f"parquet_file_count: {len(parquet_files)}",
        f"lateral_min_nm: {config.lateral_min_nm}",
        f"vertical_min_ft: {config.vertical_min_ft}",
        f"resample_rate_s: {config.resample_rate_s}",
        f"history_s: {config.history_s}",
        f"horizon_s: {config.horizon_s}",
        f"scenario_duration_s: {config.scenario_duration_s}",
        f"maneuver_conflict_prob: {config.maneuver_conflict_prob}",
        f"near_miss_scenario_prob: {config.near_miss_scenario_prob}",
        f"noise_sigma_horizontal_nm: {config.noise_sigma_horizontal_nm}",
        f"noise_sigma_vertical_ft: {config.noise_sigma_vertical_ft}",
        f"sha256: {sha256_hash}",
        f"content_sha256: {content_hash}",
    ]

    out_file = p_dir / hash_filename
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def parse_dataset_hash_file(hash_file_path: str | Path) -> dict[str, str]:
    """Parse key-value pairs from a DATASET_HASH.txt file."""
    path = Path(hash_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Hash file not found at {path}")

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def verify_dataset_hash(
    data_dir: str | Path,
    hash_filename: str = "DATASET_HASH.txt",
) -> tuple[bool, str, str]:
    """Verify that current parquet files, sizes, and file bytes match the recorded SHA-256 hashes.

    Returns
    -------
    tuple[bool, str, str]
        (matches, stored_hash, current_hash)
    """
    p_dir = Path(data_dir)
    hash_file = p_dir / hash_filename
    parsed = parse_dataset_hash_file(hash_file)

    stored_sha = parsed.get("sha256", "")
    curr_sha = compute_parquet_sha256(p_dir)
    if stored_sha != curr_sha:
        return False, stored_sha, curr_sha

    stored_content = parsed.get("content_sha256", "")
    if stored_content:
        curr_content = compute_parquet_content_sha256(p_dir)
        if stored_content != curr_content:
            return False, stored_content, curr_content

    return True, stored_sha, curr_sha
