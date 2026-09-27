# Iono Forecaster 🛰️

**AI-Based Ionospheric Scintillation Forecasting over Equatorial Africa**

> Copernicus Master's in Digital Earth — ESA Space Weather Portfolio Project  
> Author: **Samuel Appiah Kubi** | Paris Lodron University Salzburg (PLUS)

---

## Overview

IonoForecaster is an end-to-end, ESA-grade pipeline for 6-hour ahead prediction of GNSS ionospheric scintillation (S4 index) over equatorial Africa (30°W–60°E, 20°S–20°N). It uses a **Spatio-Temporal Graph Neural Network (ST-GNN)** that combines graph attention over 15 GNSS stations with bidirectional LSTM temporal encoding.

Scintillation — rapid fluctuations in GNSS signal amplitude and phase caused by ionospheric plasma irregularities — is most severe in the equatorial region, especially during post-sunset hours and geomagnetic storms. Accurate forecasting is critical for aviation, maritime navigation, precision agriculture, and land surveying across Africa.

---

## Key Features

| Feature | Detail |
|---|---|
| **Forecast horizon** | 6 hours (12 × 30-min steps), iterative recursive rollout |
| **Target region** | Equatorial Africa: 30°W–60°E, 20°S–20°N |
| **Stations** | 15 GNSS stations (IGS, AFREF, SAGAING networks) |
| **Model** | ST-GNN: GATv2Conv (4 heads) + 2-layer BiLSTM + residual decoder |
| **Input features** | 34 features: S4, TEC derivatives, solar wind, Kp/Dst, harmonics, storm phase |
| **Uncertainty** | Monte Carlo Dropout (30-sample 95% CI) |
| **Alert levels** | Watch (S4>0.3), Warning (S4>0.4), Severe (S4>0.7) |
| **ESA targets** | MAE<0.08, RMSE<0.12, F1_warning>0.85, F1_severe>0.80 |
| **Data sources** | GNSS SAMBA, IGS GIM, NOAA DSCOVR/ACE, GFZ Kp, WDC Dst, ESA Swarm, OVATION |
| **Outputs** | STAC 1.0 catalog, GeoJSON events, Folium/Plotly dashboard, alerts (JSON/CSV/Telegram/email) |

---

## Project Structure

```
iono_forecaster/
├── config.yaml                          # Master configuration
├── main.py                              # CLI orchestrator
├── requirements.txt
├── Dockerfile
├── README.md
│
├── scripts/
│   ├── utils.py                         # Logging, config, checkpointing
│   │
│   ├── downloaders/
│   │   ├── gnss_samba.py               # GNSS S4/σφ/ROTI/TEC (15 stations)
│   │   ├── igs_gim.py                  # IGS IONEX TEC maps
│   │   ├── solar_wind.py               # NOAA DSCOVR/ACE Bz, v_sw, n_p, t_p
│   │   ├── geomagnetic.py              # GFZ Kp + WDC Dst
│   │   ├── swarm.py                    # ESA Swarm EFI_LP_1B electron density
│   │   └── aurora.py                   # OVATION Prime hemispheric power
│   │
│   ├── preprocess/
│   │   ├── resample.py                 # Resampling to 30-min common grid
│   │   ├── gap_fill.py                 # Linear interpolation gap filling
│   │   ├── outlier_remove.py           # IQR-based outlier detection
│   │   └── normalise.py                # IonoScaler with physical bounds
│   │
│   ├── graph/
│   │   ├── build_graph.py              # StationGraph: distance + TEC correlation edges
│   │   └── station_coords.py           # Station metadata registry
│   │
│   ├── features/
│   │   ├── tec_derivatives.py          # ROT, ROTI
│   │   ├── coupling_functions.py       # ε, Newell rate, Ey, β_sw
│   │   ├── harmonics.py                # Sin/cos diurnal + seasonal + post-sunset flag
│   │   └── storm_time.py               # Storm onset, phase, cumulative Dst
│   │
│   ├── dataset/
│   │   ├── iono_dataset.py             # PyTorch Dataset: sliding window (T, N, F)
│   │   └── collate.py                  # Batch collation for DataLoader
│   │
│   ├── models/
│   │   ├── st_gnn.py                   # GATv2 + BiLSTM + decoder + MC Dropout
│   │   ├── losses.py                   # CombinedLoss (0.7·MSE + 0.3·MAE) + FocalLoss
│   │   └── metrics.py                  # MAE, RMSE, R², F1, POD, FAR, CSI
│   │
│   ├── train/
│   │   ├── train.py                    # Trainer: AMP, clipping, early stopping, TensorBoard
│   │   ├── temporal_cv.py              # 3-fold temporal cross-validation
│   │   └── spatial_cv.py               # Station hold-out spatial CV
│   │
│   ├── forecast/
│   │   └── iterative_rollout.py        # 12-step recursive rollout + MC Dropout ensemble
│   │
│   ├── events/
│   │   ├── detect.py                   # Event detection + GeoJSON export
│   │   └── persistence.py              # Persistence baseline + skill score
│   │
│   ├── alerts/
│   │   ├── dispatch.py                 # JSON, CSV, Telegram, SMTP alerts
│   │   └── thresholds.py               # ICAO-aligned S4 thresholds + calibration
│   │
│   ├── export/
│   │   ├── stac.py                     # STAC 1.0 Catalog + Items
│   │   └── dashboard.py                # Folium map + Plotly time series
│   │
│   └── validation/
│       └── validate.py                 # Full validation + ESA target compliance check
│
├── tests/
│   ├── test_downloaders.py
│   ├── test_preprocess.py
│   ├── test_graph.py
│   ├── test_model.py
│   └── test_forecast.py
│
└── notebooks/
    ├── 01_explore_gnss.ipynb
    ├── 02_visualise_tec.ipynb
    └── 03_analyse_scintillation.ipynb
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/appiahkubis14/iono-forecaster.git
cd iono-forecaster
pip install -r requirements.txt
```

### 2. Run full pipeline in synthetic/demo mode (no credentials needed)

```bash
python main.py --synthetic --step all
```

This runs the complete pipeline from data generation through training, forecasting, event detection, alert dispatch, validation, and dashboard generation — all with synthetic data.

### 3. Run individual steps

```bash
# Download real data (requires network access; some sources need credentials)
python main.py --step download

# Preprocess and build feature arrays
python main.py --step preprocess
python main.py --step features

# Build the station graph
python main.py --step build_graph

# Train the ST-GNN
python main.py --step train

# Resume interrupted training
python main.py --step train --resume last

# Generate 6-hour forecast
python main.py --step forecast

# Detect scintillation events
python main.py --step detect_events

# Build interactive dashboard
python main.py --step dashboard
```

### 4. Docker

```bash
# Build
docker build -t iono-forecaster .

# Run (mounts host data directory)
docker run --rm -v $(pwd)/data:/app/data iono-forecaster
```

---

## Architecture

```
Input: (B, T=12, N=15, F=34)
  │
  ├─ Input projection: Linear(F → 64)
  │
  ├─ For each timestep t:
  │   └─ SpatialEncoder (GATv2Conv × 2)
  │       ├─ GATv2(64 → 64×4heads) + ELU + LayerNorm + Dropout
  │       └─ GATv2(256 → 64) + LayerNorm
  │         → (B, N, 64) spatial embeddings
  │
  ├─ Stack temporal: (B, T, N, 64) → per-node (B·N, T, 64)
  │
  ├─ TemporalEncoder: BiLSTM(64 → 128×2) → (B·N, 256)
  │
  ├─ Decoder: Linear(256→128) + ReLU + Dropout + Linear(128→1)
  │   + residual connection
  │
  └─ Sigmoid → (B, N, 1) S4 prediction ∈ [0, 1]
```

**Iterative 6-hour forecast**: the model is called 12 times, feeding each prediction back as the S4 input feature for the next step.

**Uncertainty**: 30-sample MC Dropout → mean ± 1.96σ (95% CI).

---

## Data Sources

| Variable | Source | Frequency | Availability |
|---|---|---|---|
| S4, σφ, ROTI | GNSS SAMBA network | 1–30 min | Open |
| TEC maps | IGS IONEX | 2 hours (interpolated to 30 min) | Open |
| Bz, v_sw, n_p, t_p | NOAA DSCOVR/ACE | 1 min | Open |
| Kp index | GFZ Potsdam | 3 hours | Open |
| Dst index | WDC Kyoto | 1 hour | Open |
| Electron density | ESA Swarm EFI_LP_1B | Along-track | Requires ESA account |
| Hemispheric power | NOAA OVATION Prime | 1 min | Open |

Set `ESA_SWARM_USER` and `ESA_SWARM_PASS` environment variables for Swarm access, or use `--synthetic` flag to skip.

---

## Alert System

IonoForecaster issues three-level scintillation alerts:

| Level | S4 threshold | Meaning | Action |
|---|---|---|---|
| 🟡 **Watch** | > 0.3 | Moderate; accuracy degradation | Monitor; increase update rates |
| 🟠 **Warning** | > 0.4 | Strong; likely loss-of-lock | Switch to dual-frequency; suspend precision ops |
| 🔴 **Severe** | > 0.7 | Extreme; satellite acquisition failure | Suspend GNSS ops; use backup navigation |

Alert channels: JSON (machine-readable), CSV (log), Telegram bot, SMTP email.

Configure in `config.yaml` under `alerts:` or via environment variables:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `SMTP_SERVER`, `EMAIL_FROM`, `EMAIL_TO`

---

## Validation

```bash
python main.py --step validate
```

Outputs saved to `data/outputs/validation/`:
- `validation_report.json` — full metrics + ESA target compliance check
- `scatter_s4.png` — predicted vs observed scatter plot
- `time_series_s4.png` — time series comparison
- `event_f1.png` — F1 bar chart per alert level

### ESA Target Metrics

| Metric | Target | Rationale |
|---|---|---|
| MAE | < 0.08 S4 units | GNSS positioning error tolerance |
| RMSE | < 0.12 S4 units | Storm-event penalty |
| F1 (Warning) | > 0.85 | ICAO alert reliability requirement |
| F1 (Severe) | > 0.80 | Safety-critical event detection |

---

## Reproducibility

```bash
# Install and run tests
pytest tests/ -v --cov=scripts --cov-report=term-missing

# Expected: all tests pass; coverage > 80% on core modules
```

Random seed: `42` (configurable in `config.yaml` under `training.seed`).

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_explore_gnss.ipynb` | GNSS S4 time series exploration, seasonal patterns |
| `02_visualise_tec.ipynb` | GIM TEC map animation over equatorial Africa |
| `03_analyse_scintillation.ipynb` | Statistical analysis of storm-time scintillation |

---

## Citation

If you use IonoForecaster in your research, please cite:

```
Appiah Kubi, S. (2025). IonoForecaster: AI-Based Ionospheric Scintillation
Forecasting over Equatorial Africa using Spatio-Temporal Graph Neural Networks.
MSc thesis, Paris Lodron University Salzburg / Copernicus Master's in Digital Earth.
```

---

## Acknowledgements

- **ESA Space Weather Service Network (SWESNET)** — data products and standards
- **IGS** — IONEX TEC map archive
- **NOAA SWPC** — solar wind and geomagnetic index data
- **GFZ Potsdam** — Kp index archive
- **AFREF** — African reference GNSS network
- Supervisors and the PLUS Digital Earth programme

---

*IonoForecaster is developed as part of the ESA Copernicus Master's in Digital Earth space weather portfolio.*
