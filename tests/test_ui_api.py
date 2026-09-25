"""Regression and integration tests for AeroMetrics UI and FastAPI endpoints.

Research simulation only - not for operational use.
Verifies API routes, data fidelity, zero placeholder text, and metrics parity
with run_test_evaluation CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import torch

from src.eval.run_baseline import run_test_evaluation
from src.ui.app import app
from src.ui.service import get_torch_device, get_benchmark_metrics, _resolve_dataset_paths

client = TestClient(app)


def _has_split_scenarios(dataset: str, split: str) -> bool:
    """Return True if at least one scenario parquet file exists for the dataset and split."""
    try:
        data_dir, splits_file, _ = _resolve_dataset_paths(dataset)
        if not data_dir.exists() or not splits_file or not splits_file.exists():
            return False
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        scenario_ids = splits.get(split, [])
        return any((data_dir / f"{sid}.parquet").exists() for sid in scenario_ids)
    except Exception:
        return False


def test_system_info_endpoint():
    """Verify /api/system reports active torch device, dataset seeds, and advisory."""
    resp = client.get("/api/system")
    assert resp.status_code == 200
    data = resp.json()

    # Device must match torch.device (MPS on Apple Silicon, CUDA on GPU, CPU otherwise)
    expected_device = str(get_torch_device()).upper()
    assert data["device"] == expected_device
    assert "CUDA:0" not in data["device"] or torch.cuda.is_available()

    # Seeds must match configs/default.yaml
    assert data["seeds"]["easy"] == 42
    assert data["seeds"]["hard_v1"] == 1042
    assert data["seeds"]["hard_large"] == 2042

    # Separation minima
    assert data["separation_minima"]["lateral_nm"] == 5.0
    assert data["separation_minima"]["vertical_ft"] == 1000.0

    # Safety disclaimer
    assert "RESEARCH SIMULATION — ADVISORY ONLY — NOT FOR OPERATIONAL USE" in data["advisory_disclaimer"]


def test_datasets_endpoint():
    """Verify /api/datasets lists all 4 supported datasets."""
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    ids = [d["id"] for d in datasets]
    assert "hard_v1" in ids
    assert "hard_large" in ids
    assert "easy" in ids
    assert "opensky" in ids


def test_scenarios_listing_endpoint():
    """Verify /api/scenarios returns real scenarios for hard_v1 test split."""
    resp = client.get("/api/scenarios?dataset=hard_v1&split=test")
    assert resp.status_code == 200
    scenarios = resp.json()

    # 75 scenarios in hard_v1 test split
    assert len(scenarios) == 75

    first = scenarios[0]
    assert first["scenario_id"] == "scenario_0012"
    assert "has_conflict" in first
    assert "tag" in first
    assert "tag_type" in first
    assert "description" in first


def test_scenario_detail_endpoint():
    """Verify /api/scenario/{id} returns real non-empty tracks, predictions, and kinematics."""
    scen_file = Path("data/synthetic_hard/scenario_0012.parquet")
    if not scen_file.exists():
        pytest.skip(f"Scenario file {scen_file} not found")

    resp = client.get("/api/scenario/scenario_0012?dataset=hard_v1&split=test&model=cv_smoothed&origin=60")
    assert resp.status_code == 200
    detail = resp.json()

    assert detail["scenario_id"] == "scenario_0012"
    assert detail["dataset"] == "hard_v1"
    assert detail["split"] == "test"
    assert detail["model_name"] == "cv_smoothed"
    assert detail["origin_t"] == 60.0
    assert len(detail["available_origins"]) > 0

    # Aircraft tracks
    assert detail["aircraft_ids"] == ["AC_0", "AC_1"]
    for acid in detail["aircraft_ids"]:
        ac = detail["aircraft"][acid]
        assert len(ac["history_points"]) > 0
        assert len(ac["future_pred"]) > 0
        assert ac["altitude_ft"] > 0
        assert ac["speed_kt"] > 0
        assert 0 <= ac["heading_deg"] <= 360

    # Active pair kinematics
    pair = detail["active_pair"]
    assert pair is not None
    assert pair["curr_distance_nm"] > 0
    assert pair["predicted_cpa_nm"] >= 0
    assert pair["predicted_tau_s"] >= 0
    assert "p_risk" in pair


@pytest.mark.parametrize(
    "dataset,split,model,eval_variant,eval_model,eval_ckpt",
    [
        ("hard_v1", "test", "cv_smoothed", "hard", "cv_smoothed", None),
        ("hard_large", "test", "lstm_v1", "hard_large", "lstm", "checkpoints/lstm_v1/best.pt"),
        ("hard_v1", "val", "cv_smoothed", "hard", "cv_smoothed", None),
    ],
)
def test_metrics_endpoint_parity_with_run_baseline(
    dataset: str,
    split: str,
    model: str,
    eval_variant: str,
    eval_model: str,
    eval_ckpt: str | None,
):
    """Verify /api/metrics dynamically computes and identically matches run_test_evaluation."""
    if not _has_split_scenarios(dataset, split):
        pytest.skip(f"No scenario files found for {dataset} split '{split}'")

    # Independently compute ground truth via run_test_evaluation
    pos_metrics, conf = run_test_evaluation(
        variant=eval_variant,
        model_name=eval_model,
        checkpoint=eval_ckpt,
        split=split,
    )

    resp = client.get(f"/api/metrics?dataset={dataset}&split={split}&model={model}")
    assert resp.status_code == 200
    api_metrics = resp.json()

    # Exact sample & confusion matrix parity
    assert api_metrics["true_positives"] == conf.true_positives
    assert api_metrics["false_positives"] == conf.false_positives
    assert api_metrics["false_negatives"] == conf.false_negatives
    assert api_metrics["true_negatives"] == conf.true_negatives
    assert api_metrics["total_samples"] == conf.total_samples

    # Rate metric parity
    assert api_metrics["precision"] == pytest.approx(round(conf.precision, 4), abs=1e-4)
    assert api_metrics["recall"] == pytest.approx(round(conf.recall, 4), abs=1e-4)
    assert api_metrics["f1_score"] == pytest.approx(round(conf.f1_score, 4), abs=1e-4)
    assert api_metrics["false_alarms_per_1000_negatives"] == pytest.approx(
        round(conf.false_alarms_per_1000_negatives, 2), abs=1e-2
    )

    # Position RMSE parity
    for m in pos_metrics:
        h_str = str(m.horizon_s)
        assert h_str in api_metrics["position_rmse"]
        assert api_metrics["position_rmse"][h_str]["horizontal_nm"] == pytest.approx(
            round(m.horizontal_rmse_nm, 4), abs=1e-4
        )
        assert api_metrics["position_rmse"][h_str]["vertical_ft"] == pytest.approx(
            round(m.vertical_rmse_ft, 2), abs=1e-2
        )


@pytest.fixture(scope="module")
def real_positive_samples() -> dict[str, int]:
    """Compute true ground-truth positive-sample counts independently via run_test_evaluation."""
    samples = {}
    for ds, var in [("easy", "easy"), ("hard_v1", "hard"), ("hard_large", "hard_large")]:
        if not _has_split_scenarios(ds, "test"):
            continue
        _, conf = run_test_evaluation(variant=var, model_name="cv_smoothed", split="test")
        samples[ds] = conf.positive_samples
    return samples


@pytest.mark.parametrize("dataset", ["easy", "hard_v1", "hard_large"])
@pytest.mark.parametrize("model_name", ["cv_3step", "cv_smoothed", "lstm_v1"])
def test_no_hardcoded_numeric_shortcuts(
    dataset: str, model_name: str, real_positive_samples: dict[str, int]
):
    """Verify get_benchmark_metrics traces back to live evaluation with no shortcuts.

    Covers 3 datasets x 3 models on the test split (9 calls), proving each result's
    TP + FN strictly equals the real ground-truth positive-sample count computed
    independently via run_test_evaluation.
    """
    if not _has_split_scenarios(dataset, "test") or dataset not in real_positive_samples:
        pytest.skip(f"No scenario files found for {dataset} split 'test'")

    # Call service layer (computes live via run_test_evaluation and caches in _METRICS_CACHE)
    res = get_benchmark_metrics(dataset=dataset, split="test", model_name=model_name)

    # Prove live computation: TP + FN must strictly equal ground truth positive samples
    expected_positives = real_positive_samples[dataset]
    assert res["true_positives"] + res["false_negatives"] == expected_positives
    assert res["total_samples"] > 0
    assert res["true_positives"] > 0
    assert res["true_negatives"] > 0



def test_static_html_no_stitch_placeholders():
    """Verify served HTML contains zero mock/placeholder data from Stitch."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Mandatory title and banner
    assert "AeroMetrics Conflict Research Console" in html
    assert "RESEARCH SIMULATION — ADVISORY ONLY — NOT FOR OPERATIONAL USE" in html

    # Forbidden Stitch placeholder strings
    forbidden = [
        "SCN-8842-X7",
        "AC-104",
        "AC-219",
        "FL330|440k",
        "CUDA:0",
        "#0x4F92B",
        "#0X4F92B",
    ]
    for placeholder in forbidden:
        assert placeholder not in html, f"Found mock placeholder '{placeholder}' in static HTML"
