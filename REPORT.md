# Research Report: Air-Traffic Conflict-Risk Advisory Agent

> **CRITICAL RESEARCH NOTICE & SAFETY DISCLAIMER**:
> This software and associated models are **offline research prototypes only** and are **not intended, tested, or certified for operational air-traffic management**.
> **Under no circumstances must this software, its baseline predictors, or trained neural models be connected to any live air-traffic control (ATC), flight management system (FMS), or operational surveillance stream.**

---

## 1. Executive Summary & Problem Framing

The `atc-conflict-advisory` project investigates offline trajectory prediction and separation assurance advisory agents in dense en-route airspace. The core task evaluates whether predictive trajectory models can accurately forecast aircraft positions and provide timely, reliable conflict warnings across look-ahead horizons from 30 to 300 seconds ($t \in [30, 60, 120, 180, 240, 300]\text{ s}$).

### Operational Problem Formulation
- **Surveillance Input**: 60-second observation history (13 discrete steps at $\Delta t = 5\text{ s}$), tracking horizontal positions ($x, y$ in NM) and barometric altitude ($z$ in ft).
- **Forecast Horizon**: 300-second look-ahead (60 discrete steps at $\Delta t = 5\text{ s}$).
- **Separation Minima (ICAO/FAA En-Route Standard)**:
  $$\text{Loss of Separation (LoS)} \iff d_{\text{horizontal}}(t) < 5.0\,\text{NM} \quad \text{AND} \quad |\Delta h(t)| < 1000.0\,\text{ft}$$
- **Evaluation Paradigm**: Strict scenario-level disjoint partitioning (70% train, 15% validation, 15% test). Predictions are evaluated on two fronts:
  1. **Trajectory Position Error**: Horizontal RMSE (NM) and Vertical RMSE (ft) across forecast horizons.
  2. **Conflict Risk Advisory**: Binary classification of loss-of-separation encounters on pairwise prediction origins, evaluating detection rate, precision, recall, F1-score, false alarm rate per 1,000 negatives, first-detection lead times, and time-to-conflict (TTC) mean absolute error.

---

## 2. Datasets & Surveillance Pipelines

### 2.1 Synthetic Datasets (`easy`, `hard_v1`, `hard_large`)

To systematically benchmark predictive architectures, three synthetic benchmarks were developed:

| Dataset | Scenarios | Seed | Description | Label Audit Findings |
|:---|:---:|:---:|:---|:---|
| **`easy`** | 500 | 42 | Pure straight-line constant-velocity cruise; no maneuvers; noise-free surveillance. | Baseline scored recall 1.0, F1 0.9896, TTC MAE 0.00 s. Too trivial for model discrimination. |
| **`hard_v1`** | 500 | 1042 | ~50% maneuvering conflicts (standard-rate turns $\pm 15^\circ\text{--}45^\circ$, climbs/level-offs $\pm 1000\text{--}2500\text{ fpm}$), near-miss encounters (5.0–8.0 NM, 1000–1500 ft), and Gaussian observation noise ($\sigma_{\text{lat}} = 0.05\text{ NM}, \sigma_{\text{alt}} = 30\text{ ft}$) applied strictly to history. | 150 confirmed conflict scenarios; 0 accidental conflicts in clean scenarios. Frozen with content SHA-256 in `DATASET_HASH.txt`. |
| **`hard_large`** | 3,000 | 2042 | Scaled hard benchmark with identical maneuver, noise, and encounter dynamics under an independent seed. Partitioned into 2,100 train / 450 val / 450 test scenarios. | 900 confirmed conflict scenarios (146 distinct conflict pairs in test split); 0 accidental conflicts in clean scenarios. Frozen in `DATASET_HASH.txt`. |

### 2.2 OpenSky Network Real ADS-B Surveillance Data

To validate the simulation-trained models against real-world flight dynamics, empirical ADS-B surveillance data was gathered from the OpenSky Network for the Central European En-Route Sector ($49.0^\circ\text{--}52.0^\circ\text{ N}, 7.0^\circ\text{--}11.0^\circ\text{ E}$, barometric altitude $> 3,000\text{ m}$ / FL100):

1. **Historical Access Audit**:
   - The OpenSky REST history API (`https://opensky-network.org/api/states/all?time=...`) returned `HTTP 403 Forbidden` (`b'Authenticate to get historical data'`). OpenSky restricts historical REST access to registered, approved academic accounts.
   - An unauthenticated S3 scientific sample archive (`states_2017-06-05-00.csv.tar`) was initially retrieved, but per rigorous data governance protocols, was discarded because it was not accessed through OpenSky's primary authenticated Trino interface.
2. **Live Polling Streaming Collection**:
   - The confirmed public live endpoint (`https://opensky-network.org/api/states/all?lamin=49.0&lamax=52.0&lomin=7.0&lomax=11.0`) was polled continuously once every 12 seconds for 20 minutes (95 sequential polls).
   - **Live Yield**: 5,359 state vector observations collected across 118 distinct aircraft tracks.
   - **En-Route Trajectory Reassembly**: 84 flights maintained continuous, uninterrupted surveillance for $\ge 6\text{ minutes}$ ($360\text{ s}$), yielding **985 prediction windows** (60 s history + 300 s future horizon) projected onto the local Cartesian plane.

---

## 3. Predictive Methodology

### 3.1 Constant-Velocity Baselines
- **`cv_3step`**: Backward finite-difference velocity estimation over the most recent 3 steps ($t=50, 55, 60\text{ s}$). Sensitive to sensor observation noise.
- **`cv_smoothed`**: Ordinary least-squares linear regression fit across the full 13-step ($60\text{ s}$) observation window for $[x, y, z]$. Filters high-frequency observation noise and provides the reference kinematic extrapolation.

### 3.2 Residual LSTM Architecture
- **Model Design**: 2-layer Recurrent Neural Network (128 hidden units) consuming the 13-step history sequence $[\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t, \Delta \mathbf{x}_t, \Delta \mathbf{y}_t, \Delta \mathbf{z}_t]$.
- **Residual Formulation**: The LSTM outputs multi-horizon displacement corrections:
  $$\hat{\mathbf{p}}_{t+\tau} = \mathbf{p}^{\text{cv\_smoothed}}_{t+\tau} + \Delta \mathbf{p}^{\text{LSTM}}_{t+\tau}$$
- **Zero-Initialization Contract**: The output linear projection is initialized with zero weights and biases, guaranteeing that at epoch 0, the model's predictions exactly match `cv_smoothed`.
- **Training Setup**: Trained on `hard_large` train split using AdamW, MSE trajectory loss, gradient clipping (1.0), and early stopping on validation loss (checkpoint: `checkpoints/lstm_v1/best.pt`).

---

## 4. Empirical Evaluation Results

### 4.1 Trajectory Position Prediction Error

#### Table 1: Horizontal and Vertical Position RMSE on `hard_large` TEST ($N=64,902$ windows) and `hard_v1` TEST ($N=10,382$ windows)

| Horizon | `cv_smoothed` (`hard_large`) | Residual LSTM (`hard_large`) | `cv_smoothed` (`hard_v1`) | Residual LSTM (`hard_v1`) |
|:---:|:---:|:---:|:---:|:---:|
| **30 s** | 0.1690 NM / 65.72 ft | **0.0979 NM** / **34.53 ft** | 0.1651 NM / 64.15 ft | **0.0953 NM** / **33.71 ft** |
| **60 s** | 0.3545 NM / 108.15 ft | **0.2526 NM** / **59.40 ft** | 0.3480 NM / 103.90 ft | **0.2465 NM** / **56.86 ft** |
| **120 s** | 0.8359 NM / 189.22 ft | **0.6907 NM** / **93.98 ft** | 0.8267 NM / 179.77 ft | **0.6824 NM** / **87.89 ft** |
| **180 s** | 1.4344 NM / 267.48 ft | **1.2555 NM** / **117.78 ft** | 1.4234 NM / 253.25 ft | **1.2462 NM** / **109.73 ft** |
| **240 s** | 2.1312 NM / 344.94 ft | **1.9233 NM** / **135.33 ft** | 2.1167 NM / 326.32 ft | **1.9109 NM** / **125.81 ft** |
| **300 s** | 2.9138 NM / 421.58 ft | **2.6798 NM** / **148.90 ft** | 2.8899 NM / 398.66 ft | **2.6582 NM** / **138.63 ft** |

---

### 4.2 Conflict Prediction & Advisory Performance

#### Table 2: Conflict Advisory Classification Metrics on `hard_large` TEST (450 scenarios, 155,498 sample pairs, 95% Scenario-Level Bootstrap CIs)

| Metric | `cv_smoothed` Baseline | Residual LSTM (`best.pt`) |
|:---|:---:|:---:|
| **Conflict Detection Rate** | 1.0000 [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] |
| **True Positives (TP)** | 1,378 | **1,427** |
| **False Positives (FP)** | 56 | **6** |
| **False Negatives (FN)** | 217 | **168** |
| **True Negatives (TN)** | 153,847 | **153,897** |
| **Sample-Level Precision** | 0.9609 [0.9435, 0.9737] | **0.9958** [0.9904, 0.9993] |
| **Sample-Level Recall** | 0.8639 [0.8248, 0.8988] | **0.8947** [0.8616, 0.9259] |
| **Sample-Level F1-Score** | 0.9099 [0.8852, 0.9307] | **0.9425** [0.9238, 0.9594] |
| **False Alarms / 1000 Neg** | 0.36 | **0.04** |
| **FP Breakdown (Category)** | 6 injected / 50 near-miss / 0 other | **2 injected / 4 near-miss / 0 other** |
| **FP Breakdown (Cause)** | 8 lateral / 48 vertical / 0 both | **6 lateral / 0 vertical / 0 both** |
| **Maneuvering Recall** | 0.7206 ($557 / 773$) | **0.7865** ($608 / 773$) |
| **Non-Maneuvering Recall** | **0.9988** ($821 / 822$) | 0.9964 ($819 / 822$) |
| **Mean Absolute TTC Error** | **0.94 s** | 1.03 s |

> **Note on `cv_3step` Baseline (Pre-Upgrade Run)**:
> The un-smoothed 3-step constant-velocity baseline (`cv_3step`) was evaluated on `hard_large` TEST with the earlier metric definitions in `task-816.log` (lines 185–240), where it yielded:
> Conflict Detection Rate: 0.9863 (144 / 146, 95% CI: [0.9655, 1.0000]); TP=872, FP=333, FN=723, TN=153,570; Precision: 0.7237 [0.6675, 0.7738]; Recall: 0.5467 [0.5237, 0.5692]; F1: 0.6229 [0.5942, 0.6476]; False Alarms / 1000 Neg: 2.16; FP Breakdown: 31 injected / 275 near-miss / 27 other clean (cause: 23 lateral / 299 vertical / 11 both); Maneuvering Recall: 0.5201 (TP=402, FN=371); Non-Maneuvering Recall: 0.5718 (TP=470, FN=352); Mean Absolute TTC Error: 2.23 s. Because `cv_3step` was not re-evaluated after the A2 evaluation upgrades (lead-time filters and zero-positive formatting), it is excluded from Table 2 and cited here as a pre-upgrade baseline reference.

---

### 4.3 OpenSky Real ADS-B Trajectory Validation

#### Table 3: Side-by-Side Position RMSE: Real Live OpenSky vs Synthetic `hard_large` TEST

| Horizon | `cv_smoothed` (Live OpenSky) | Residual LSTM (Live OpenSky) | LSTM Gain (Live OpenSky) | `cv_smoothed` (`hard_large` TEST) | Residual LSTM (`hard_large` TEST) | LSTM Gain (`hard_large` Synthetic) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **30 s** | 0.2723 NM / 199.39 ft | 0.2691 NM / 386.45 ft | **+1.2%** Horiz / <span style="color:red">**-93.8% Vert**</span> | 0.1690 NM / 65.72 ft | 0.0979 NM / 34.53 ft | **+42.1%** Horiz / **+47.5% Vert** |
| **60 s** | 0.5835 NM / 396.31 ft | 0.5526 NM / 867.98 ft | **+5.3%** Horiz / <span style="color:red">**-119.0% Vert**</span> | 0.3545 NM / 108.15 ft | 0.2526 NM / 59.40 ft | **+28.7%** Horiz / **+45.1% Vert** |
| **120 s** | 1.4043 NM / 882.32 ft | 1.3448 NM / 1806.65 ft | **+4.2%** Horiz / <span style="color:red">**-104.8% Vert**</span> | 0.8359 NM / 189.22 ft | 0.6907 NM / 93.98 ft | **+17.4%** Horiz / **+50.3% Vert** |
| **180 s** | 2.4279 NM / 1455.18 ft | 2.3293 NM / 2695.49 ft | **+4.1%** Horiz / <span style="color:red">**-85.2% Vert**</span> | 1.4344 NM / 267.48 ft | 1.2555 NM / 117.78 ft | **+12.5%** Horiz / **+56.0% Vert** |
| **240 s** | 3.6044 NM / 2105.77 ft | 3.4685 NM / 3540.62 ft | **+3.8%** Horiz / <span style="color:red">**-68.1% Vert**</span> | 2.1312 NM / 344.94 ft | 1.9233 NM / 135.33 ft | **+9.8%** Horiz / **+60.8% Vert** |
| **300 s** | 4.9082 NM / 2778.97 ft | 4.7339 NM / 4334.14 ft | **+3.6%** Horiz / <span style="color:red">**-56.0% Vert**</span> | 2.9138 NM / 421.58 ft | 2.6798 NM / 148.90 ft | **+8.0%** Horiz / **+64.7% Vert** |

---

## 5. Explicit Limitations & Failure Analysis

In compliance with scientific integrity and safety requirements, the following limitations are explicitly noted:

1. **Failure of Vertical Generalization to Real-World ADS-B**:
   The residual LSTM's learned vertical corrections **did not generalize** to the empirical OpenSky live sample. At every single reported horizon, the LSTM's vertical RMSE was significantly worse than the linear constant-velocity baseline (`cv_smoothed`). For example, at the 60 s horizon, the LSTM produced **867.98 ft** vertical RMSE compared to **396.31 ft** for `cv_smoothed` (+119.0% error inflation), and at 300 s produced **4,334.14 ft** vs **2,778.97 ft**. Meanwhile, lateral prediction gains shrank from +28.7% in simulation to roughly on par (+5.3%) in real flight. This failure stems from synthetic training on idealized piecewise-linear altitude profiles, which led the model to predict spurious vertical accelerations on real-world flights undergoing barometric drift and operational flight-level changes.
2. **Small Scale of Empirical ADS-B Sample**:
   The live OpenSky evaluation dataset consists of a single 20-minute live polling session ($N=985$ windows across 84 qualifying flights). While completely genuine and un-synthesized, this sample is modest in scale and cannot substitute for an authenticated multi-day historical repository covering varied weather, convective turbulence, and diverse airspace geometries.
3. **Historical Subset Discrepancy Note**:
   During benchmark development, an unresolved discrepancy occurred once between two logged runs of a subset-count metric ($N=25,769$ / $\text{TP}=555$ vs $N=24,273$ / $\text{TP}=557$). The latter figure ($N=24,273$ / $\text{TP}=557$) was reproduced independently across all subsequent runs and test executions, and is what is officially reported. However, the exact mechanical origin of the earlier logged figure was not conclusively established.
4. **Non-Operational Prototype Status**:
   This software is strictly an academic research prototype. It does not account for transponder loss, primary radar blending, latency jitter, or human-machine interface safety factors, and **must never be connected to live operational air-traffic systems**.
