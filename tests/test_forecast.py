"""
tests/test_forecast.py — IonoForecaster
Tests for iterative forecast, event detection, persistence baseline.

Author: Samuel Appiah Kubi
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


N_NODES = 5
N_FEATURES = 10
T = 12


@pytest.fixture
def minimal_cfg(tmp_path):
    return {
        "temporal": {"resolution_minutes": 30, "start_date": "2023-01-01",
                     "end_date": "2023-01-07", "train_end": "2023-01-05",
                     "val_end": "2023-01-06"},
        "model": {"input_timesteps": T},
        "forecast": {"horizon_steps": 6, "mc_dropout_samples": 5},
        "paths": {
            "outputs_dir": str(tmp_path / "outputs"),
            "models_dir": str(tmp_path / "models"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "events": {
            "thresholds": {"watch": 0.3, "warning": 0.4, "severe": 0.7},
            "min_duration_minutes": 30,
        },
    }


@pytest.fixture
def graph_tensors():
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    import torch
    adj = torch.eye(N_NODES)
    src, dst = torch.meshgrid(torch.arange(N_NODES), torch.arange(N_NODES), indexing="ij")
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight, adj


def test_event_detection_basic(minimal_cfg):
    from scripts.events.detect import detect_events

    n_ts = 48
    station_ids = ["ACCR", "LAGO"]
    records = []
    timestamps = pd.date_range("2023-01-01", periods=n_ts, freq="30min")

    for sid in station_ids:
        s4 = np.random.uniform(0, 0.3, n_ts)
        # Inject a warning event at timesteps 10-20
        s4[10:20] = 0.6
        for ts, val in zip(timestamps, s4):
            records.append({"timestamp": ts, "station_id": sid, "s4_forecast": val})

    fc_df = pd.DataFrame(records)
    events_df = detect_events(fc_df, min_duration_minutes=30)
    # Should detect at least 1 event per station
    assert len(events_df) >= 2


def test_event_detection_no_events(minimal_cfg):
    from scripts.events.detect import detect_events

    timestamps = pd.date_range("2023-01-01", periods=24, freq="30min")
    records = [{"timestamp": ts, "station_id": "ACCR", "s4_forecast": 0.1}
               for ts in timestamps]
    fc_df = pd.DataFrame(records)
    events_df = detect_events(fc_df)
    assert len(events_df) == 0


def test_persistence_forecast_shape():
    from scripts.events.persistence import persistence_forecast

    T_data, N = 50, 5
    s4 = np.random.uniform(0, 1, (T_data, N)).astype(np.float32)
    preds = persistence_forecast(s4, forecast_steps=6)
    assert preds.shape == (T_data, 6, N)


def test_persistence_evaluate():
    from scripts.events.persistence import evaluate_persistence

    T_data, N = 100, 3
    s4 = np.random.uniform(0, 0.5, (T_data, N)).astype(np.float32)
    results = evaluate_persistence(s4, forecast_steps=3)
    assert "step_1" in results
    assert "mae" in results["step_1"]
    assert results["step_1"]["mae"] >= 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_iterative_forecast_output(minimal_cfg, graph_tensors):
    import torch
    from scripts.models.st_gnn import STGNN
    from scripts.forecast.iterative_rollout import IterativeForecast

    model = STGNN(n_features=N_FEATURES, n_nodes=N_NODES,
                  spatial_hidden=16, spatial_out=16, n_heads=2,
                  lstm_hidden=16, lstm_layers=1, decoder_hidden=16)

    edge_index, edge_weight, adj = graph_tensors
    forecaster = IterativeForecast(model, edge_index, edge_weight, adj, minimal_cfg)

    x_init = torch.randn(1, T, N_NODES, N_FEATURES)
    timestamps = pd.date_range("2023-01-01", periods=T, freq="30min")
    station_ids = [f"S{i}" for i in range(N_NODES)]

    df = forecaster.forecast(x_init, timestamps, station_ids)
    assert len(df) == 6 * N_NODES  # 6 steps × 5 stations
    assert "s4_forecast" in df.columns
    assert "s4_lower" in df.columns
    assert "s4_upper" in df.columns
    assert df["s4_forecast"].between(0, 1).all()


def test_alert_thresholds_classify():
    from scripts.alerts.thresholds import classify_s4
    assert classify_s4(0.1) == "quiet"
    assert classify_s4(0.35) == "watch"
    assert classify_s4(0.5) == "warning"
    assert classify_s4(0.8) == "severe"
