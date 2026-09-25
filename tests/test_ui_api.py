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
from src.ui.service import get_torch_device

client = TestClient(app)


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


def test_metrics_endpoint_parity_with_run_baseline():
    """Verify /api/metrics identically matches run_test_evaluation CLI output for hard_v1 test."""
    resp = client.get("/api/metrics?dataset=hard_v1&split=test&model=cv_smoothed")
    assert resp.status_code == 200
    api_metrics = resp.json()

    # Compare with known hard_v1 TEST cv_smoothed evaluation
    assert round(api_metrics["precision"], 4) == 0.9107
    assert round(api_metrics["recall"], 4) == 0.8226
    assert round(api_metrics["f1_score"], 4) == 0.8644
    assert api_metrics["true_positives"] == 153
    assert api_metrics["false_positives"] == 15
    assert api_metrics["false_negatives"] == 33
    assert api_metrics["true_negatives"] == 23840
    assert api_metrics["total_samples"] == 24041
    assert round(api_metrics["false_alarms_per_1000_negatives"], 2) == 0.63
    assert round(api_metrics["tp_time_to_conflict_mae_s"], 2) == 0.92

    # Also test lstm_v1 model switching
    resp_lstm = client.get("/api/metrics?dataset=hard_v1&split=test&model=lstm_v1")
    assert resp_lstm.status_code == 200
    lstm_metrics = resp_lstm.json()
    assert round(lstm_metrics["precision"], 4) == 1.0000
    assert round(lstm_metrics["recall"], 4) == 0.8280
    assert round(lstm_metrics["f1_score"], 4) == 0.9059
    assert lstm_metrics["true_positives"] == 154
    assert lstm_metrics["false_positives"] == 0


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
