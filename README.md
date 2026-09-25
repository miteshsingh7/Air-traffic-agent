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

## 6. Key Research Findings & Limitations

For complete benchmark tables, bootstrap confidence intervals, and failure analysis, see [`REPORT.md`](file:///Users/miteshsingh/Documents/projects/Air-Traffic%20Conflict-Risk%20Advisory%20Agent/REPORT.md).

- **Synthetic Performance**: On `hard_large` TEST, the residual LSTM reduced false alarms from 56 down to 6 (+99.6% precision, 0.9230 F1) and reduced 300 s position RMSE from 2.91 NM / 421.6 ft down to 2.68 NM / 148.9 ft.
- **Empirical ADS-B Generalization Failure**: When tested on real OpenSky flights, the LSTM's vertical error **more than doubled** relative to `cv_smoothed` (867.98 ft vs 396.31 ft at 60 s; 4,334.14 ft vs 2,778.97 ft at 300 s), while lateral improvements shrank to +1.2% to +5.3%.
- **Safety Restriction**: This repository represents an offline research prototype only and must never be connected to live operational air-traffic management systems.
