# Air-Traffic Conflict-Risk Advisory Agent (`atc-conflict-advisory`)

> **CRITICAL RESEARCH NOTICE & SAFETY DISCLAIMER**:
> This software is for **research simulation only** and is **not for operational use**. It is an offline, advisory-only research prototype. **Never connect this software, its baseline predictors, or trained neural models to any live air-traffic control (ATC) or operational flight-management system.**

---

## 1. Overview

The `atc-conflict-advisory` project is an offline research benchmark for investigating trajectory prediction, separation monitoring, and loss-of-separation (LoS) risk advisory algorithms in high-density airspace. The framework supports synthetic scenario generation across calibrated complexity tiers, geometric ground-truth auditing, content-addressed dataset hashing, kinematic baselines, residual recurrent neural networks, and live empirical ADS-B surveillance ingestion via the OpenSky Network.

---

## 2. Airspace & Separation Standards

Configuration parameters are governed by `configs/default.yaml`:
- **Lateral Separation Minimum**: 5.0 Nautical Miles (NM)
- **Vertical Separation Minimum**: 1,000 Feet (ft)
- **Sampling Interval**: 5 seconds ($\Delta t = 5\text{ s}$)
- **Surveillance Observation History**: 60 seconds (13 discrete timesteps)
- **Look-Ahead Forecast Horizon**: 300 seconds (60 discrete timesteps)
- **Prediction Origins**: Evaluated every 30 seconds from $t=60\text{ s}$ to $t=900\text{ s}$
- **Airspace Bounds**: $200 \times 200\text{ NM}$ horizontal domain, 10,000 ft to 40,000 ft altitude

A conflict (Loss of Separation) occurs at time $t$ if and only if:
$$\text{lateral\_distance}(t) < 5.0\,\text{NM} \quad \text{AND} \quad |\Delta h(t)| < 1000.0\,\text{ft}$$

---

## 3. Repository Structure

```
atc-conflict-advisory/
├── configs/
│   └── default.yaml             # Separation minima, horizons, and variant settings
├── data/
│   ├── synthetic_hard/          # Frozen hard_v1 dataset (500 scenarios, seed 1042)
│   ├── synthetic_hard_large/    # Frozen hard_large dataset (3,000 scenarios, seed 2042)
│   └── opensky_live_sample/     # Reassembled real ADS-B flights from live polling
├── checkpoints/
│   └── lstm_v1/best.pt          # Trained Residual LSTM checkpoint
├── src/
│   ├── conflict/
│   │   └── geometry.py          # Vectorized pairwise loss-of-separation geometry
│   ├── data/
│   │   ├── dataset_hash.py      # Content-addressed SHA-256 dataset integrity hashing
│   │   ├── live_collector.py    # Live OpenSky REST API polling & track reassembly
│   │   ├── splits.py            # Scenario-level disjoint train/val/test split generator
│   │   ├── synthetic.py         # Kinematic trajectory generator with maneuver injection
│   │   └── windows.py           # Sliding history/future window extractor
│   ├── eval/
│   │   ├── baseline_diagnostic.py # Baseline exactness diagnostic
│   │   ├── conflict_eval.py     # Pairwise conflict detection & bootstrap CI metrics
│   │   ├── label_audit.py       # Geometric ground-truth label audit
│   │   ├── metrics.py           # Multi-horizon trajectory position RMSE (NM / ft)
│   │   ├── run_baseline.py      # Test split benchmark runner (cv_3step, cv_smoothed, lstm)
│   │   └── run_opensky_eval.py  # Position RMSE evaluation on empirical OpenSky data
│   └── models/
│       ├── baseline.py          # 3-step finite difference & smoothed CV predictors
│       ├── lstm_model.py        # PyTorch 2-layer residual LSTM architecture
│       ├── lstm_predictor.py    # Zero-initialized residual trajectory inference wrapper
│       └── train_lstm.py        # PyTorch training loop with validation early stopping
├── tests/                       # Comprehensive pytest suite (42 unit & integration tests)
├── REPORT.md                    # Consolidated research findings & limitations
└── README.md                    # Project documentation & execution guide
```

---

## 4. Setup and Installation

Requirements: Python 3.10+, PyTorch, NumPy, Pandas, PyArrow, PyYAML, SciPy.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Execution Guide

### 5.1 Running Unit & Regression Tests

Run all 42 tests in the suite:
```bash
pytest -v
```

### 5.2 Synthetic Data Generation & Integrity Hashing

Generate synthetic datasets across calibrated variants:
```bash
# 1. Easy dataset (500 scenarios, straight-line, seed 42)
python3 -m src.data.synthetic --variant easy --num-scenarios 500

# 2. Hard v1 dataset (500 scenarios, maneuvering + near-miss + noise, seed 1042)
python3 -m src.data.synthetic --variant hard --num-scenarios 500

# 3. Hard Large benchmark (3000 scenarios, seed 2042)
python3 -m src.data.synthetic --variant hard_large --num-scenarios 3000
```

Verify content-addressed SHA-256 hashes against frozen digests:
```bash
python3 -m src.data.dataset_hash --data-dir data/synthetic_hard --verify
python3 -m src.data.dataset_hash --data-dir data/synthetic_hard_large --verify
```

### 5.3 Geometric Label Auditing

Audit trajectory parquet files against mathematical loss-of-separation geometry:
```bash
python3 -m src.eval.label_audit --data-dir data/synthetic_hard_large
```

### 5.4 Evaluating Baselines on Test Partition

Evaluate kinematic predictors on the isolated `hard_large` test split:
```bash
# 3-step constant-velocity finite difference baseline
python3 -m src.eval.run_baseline --variant hard_large --model cv_3step --split test

# 13-step smoothed linear regression baseline
python3 -m src.eval.run_baseline --variant hard_large --model cv_smoothed --split test
```

### 5.5 Training & Evaluating the Residual LSTM

Train the residual displacement LSTM on `hard_large` train split:
```bash
python3 -m src.models.train_lstm --variant hard_large --epochs 40 --batch-size 256
```

Evaluate the trained checkpoint on the test split:
```bash
python3 -m src.eval.run_baseline --variant hard_large --model lstm --checkpoint checkpoints/lstm_v1/best.pt --split test
```

### 5.6 OpenSky Real ADS-B Surveillance Collection & Validation

Collect live state vectors over the Central European en-route sector ($49.0^\circ\text{--}52.0^\circ\text{ N}, 7.0^\circ\text{--}11.0^\circ\text{ E}$, $> \text{FL100}$):
```bash
# Poll live endpoint every 12 seconds for 20 minutes and reassemble continuous tracks
python3 -u -m src.data.live_collector --duration-minutes 20.0 --interval 12.0 --output-dir data/opensky_live_sample
```

Evaluate baseline and LSTM position prediction errors on the live-collected sample:
```bash
python3 -m src.eval.run_opensky_eval --data-dir data/opensky_live_sample --checkpoint checkpoints/lstm_v1/best.pt
```

---

## 6. Research Console (Web UI)

The **AeroMetrics Conflict Research Console** is an offline, data-driven single-page research workstation for visualizing high-density encounters, evaluating loss-of-separation (LoS) prediction performance, and exploring tactical conflict resolution advisories. It faithfully implements the research dashboard specification with zero mock or placeholder figures.

> **CRITICAL ADVISORY DISCLAIMER**:
> The Research Console displays a persistent amber banner on every view:
> **RESEARCH SIMULATION — ADVISORY ONLY — NOT FOR OPERATIONAL USE**.
> It is an offline evaluation platform and must never be connected to operational ATC infrastructure.

### 6.1 Launching the Console

Start the Uvicorn ASGI server hosting the FastAPI backend:
```bash
python3 -m uvicorn src.ui.app:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### 6.2 REST API Specification

All backend endpoints are served under `/api/`:

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/system` | `GET` | None | Returns active PyTorch compute device (`MPS`, `CUDA`, `CPU`), dataset random seeds, separation minima (5.0 NM, 1000 ft), observation/look-ahead horizons, and safety disclaimer. |
| `/api/datasets` | `GET` | None | Lists available datasets (`hard_v1`, `hard_large`, `easy`, `opensky`) with scenario counts, disk sizes, and split manifests. |
| `/api/scenarios` | `GET` | `dataset` (str), `split` (str, default `test`) | Enumerates scenarios in the given split with conflict presence flags, trajectory tags (e.g., `MANEUVERING`, `NEAR_MISS`, `NOMINAL`), and aircraft counts. |
| `/api/scenario/{id}` | `GET` | `dataset` (str), `split` (str), `model` (str, default `cv_smoothed`), `origin` (float, default `60.0`) | Returns comprehensive per-scenario discrete telemetry (t, ACID, altitude, GS, heading, ROCD, x, y), predicted future positions across all aircraft, pairwise closest point of approach (CPA), time-to-CPA ($\tau$), closure rate, and collision risk index $P(\text{risk})$. |
| `/api/metrics` | `GET` | `dataset` (str), `split` (str), `model` (str) | Computes and returns multi-horizon trajectory position RMSE (30s, 60s, 120s, 180s, 240s, 300s) and pairwise conflict classification performance (TP, FP, FN, TN, precision, recall, F1, false alarms per 1,000 negatives, lead-time distribution, and TTC MAE). |

### 6.3 Live Metric Computation Guarantee

**All metrics returned by `/api/metrics` are computed live** via `run_test_evaluation()` from `src.eval.run_baseline`—the exact same evaluation engine used by the command-line benchmarks. There are **zero hardcoded metric dictionaries or shortcut lookups**. Computed results are cached in-memory solely to avoid redundant recalculation across rapid UI view switches.

---

## 7. Key Research Findings & Limitations

For complete benchmark tables, bootstrap confidence intervals, and failure analysis, see [`REPORT.md`](file:///Users/miteshsingh/Documents/projects/Air-Traffic%20Conflict-Risk%20Advisory%20Agent/REPORT.md).

- **Synthetic Performance**: On `hard_large` TEST, the residual LSTM reduced false alarms from 56 down to 6 (+99.6% precision, 0.9230 F1) and reduced 300 s position RMSE from 2.91 NM / 421.6 ft down to 2.68 NM / 148.9 ft.
- **Empirical ADS-B Generalization Failure**: When tested on real OpenSky flights, the LSTM's vertical error **more than doubled** relative to `cv_smoothed` (867.98 ft vs 396.31 ft at 60 s; 4,334.14 ft vs 2,778.97 ft at 300 s), while lateral improvements shrank to +1.2% to +5.3%.
- **Safety Restriction**: This repository represents an offline research prototype only and must never be connected to live operational air-traffic management systems.
