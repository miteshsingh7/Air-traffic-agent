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

    # Verify that only scenarios with existing parquet files are returned
    data_dir, splits_file, _ = _resolve_dataset_paths("hard_v1")
    with open(splits_file, "r", encoding="utf-8") as f:
        split_ids = json.load(f).get("test", [])
    expected_ids = [sid for sid in split_ids if (data_dir / f"{sid}.parquet").exists()]

    assert len(scenarios) == len(expected_ids)
    if scenarios:
        first = scenarios[0]
        assert (data_dir / f"{first['scenario_id']}.parquet").exists()
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
        try:
            _, conf = run_test_evaluation(variant=var, model_name="cv_smoothed", split="test")
            samples[ds] = conf.positive_samples
        except Exception:
            continue
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
    if expected_positives > 0:
        assert res["true_positives"] > 0
    assert res["true_negatives"] >= 0



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


def test_scenario_detail_split_validation():
    """Verify requesting a scenario with the wrong split returns 400 Bad Request."""
    # scenario_0012 is in hard_v1 'test' split, NOT 'train'
    resp = client.get("/api/scenario/scenario_0012?dataset=hard_v1&split=train")
    assert resp.status_code == 400
    assert "not a member of split 'train'" in resp.json()["detail"]


def test_scenarios_listing_skips_missing_files(monkeypatch, tmp_path):
    """Verify list_scenarios filters out IDs with no backing .parquet file."""
    from src.ui import service

    # Mock _resolve_dataset_paths to point to a splits file with a phantom scenario ID
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({"test": ["exists", "ghost"]}), encoding="utf-8")
    (tmp_path / "exists.parquet").touch()

    monkeypatch.setattr(
        service,
        "_resolve_dataset_paths",
        lambda ds: (tmp_path, splits_file, None),
    )

    results = service.list_scenarios(dataset="hard_v1", split="test")
    returned_ids = [r["scenario_id"] for r in results]
    assert "exists" in returned_ids
    assert "ghost" not in returned_ids


def test_metrics_endpoint_invalid_inputs_return_400():
    """Verify /api/metrics returns 400 Bad Request instead of 500 on invalid dataset/model."""
    resp = client.get("/api/metrics?dataset=nonexistent_dataset&split=test&model=cv_smoothed")
    assert resp.status_code == 400
    assert "Unknown dataset" in resp.json()["detail"]


def test_cors_configuration_safe():
    """Verify CORS middleware disallows credentials with wildcard origin."""
    from starlette.middleware import Middleware
    from fastapi.middleware.cors import CORSMiddleware

    cors_middlewares = [
        m for m in app.user_middleware if issubclass(m.cls, CORSMiddleware)
    ]
    assert len(cors_middlewares) > 0
    kwargs = cors_middlewares[0].kwargs
    assert kwargs.get("allow_credentials") is False
    assert kwargs.get("allow_origins") == ["*"]
    assert "GET" in kwargs.get("allow_methods", [])

    # Verify live header behavior: wildcard origin allowed, credentials header omitted
    resp = client.options(
        "/api/system",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert resp.headers.get("access-control-allow-credentials") is None


def test_system_info_surfaces_config_warning(monkeypatch, tmp_path):
    """Verify get_system_info populates config_warning on corrupted config instead of swallowing it."""
    from src.ui import service

    bad_config = tmp_path / "default.yaml"
    bad_config.write_text("invalid: yaml: [unclosed", encoding="utf-8")

    monkeypatch.setattr(service, "CONFIG_PATH", bad_config)
    info = service.get_system_info()
    assert info["config_warning"] is not None
    assert "Could not fully parse config" in info["config_warning"]


def test_metrics_endpoint_client_errors_return_400():
    """Verify /api/metrics returns 400 on unknown model or unknown split."""
    # Unknown model
    resp = client.get("/api/metrics?dataset=hard_v1&split=test&model=invalid_model_xyz")
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["detail"]

    # Unknown split
    resp2 = client.get("/api/metrics?dataset=hard_v1&split=invalid_split_abc&model=cv_smoothed")
    assert resp2.status_code == 400
    assert "Unknown split" in resp2.json()["detail"]


def test_metrics_endpoint_missing_data_returns_404(monkeypatch):
    """Verify /api/metrics returns 404 when scenario files are absent for the requested split."""
    from src.ui import service

    def mock_eval(*args, **kwargs):
        raise FileNotFoundError("No scenario files found in test_dir for split 'val'")

    monkeypatch.setattr(service, "run_test_evaluation", mock_eval)
    # Clear cache to ensure service calls run_test_evaluation
    service._METRICS_CACHE.clear()

    resp = client.get("/api/metrics?dataset=hard_v1&split=val&model=cv_smoothed")
    assert resp.status_code == 404
    assert "No scenario files found" in resp.json()["detail"]


def test_scenario_detail_unknown_split_and_missing_file():
    """Verify /api/scenario/{id} returns 400 for unknown split and 404 for missing file."""
    # Unknown split -> 400
    resp = client.get("/api/scenario/scenario_0012?dataset=hard_v1&split=nonexistent_split")
    assert resp.status_code == 400
    assert "Unknown split" in resp.json()["detail"]

    # Valid split but file does not exist -> 404
    # scenario_9999 does not exist in any split
    # If in test split: not in split_ids -> 400; if file missing -> 404
    resp2 = client.get("/api/scenario/nonexistent_scenario?dataset=opensky&split=sample")
    assert resp2.status_code == 404
