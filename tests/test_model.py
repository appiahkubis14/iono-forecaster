"""
tests/test_model.py — IonoForecaster
Unit tests for the ST-GNN model, losses, and metrics.

Author: Samuel Appiah Kubi
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


N_NODES = 5
N_FEATURES = 10
T = 12
B = 4


@pytest.fixture
def graph_tensors():
    """Minimal dense graph tensors for testing (no PyG required)."""
    import torch
    adj = torch.ones(N_NODES, N_NODES)
    torch.diagonal(adj).fill_(1.0)
    adj = adj / adj.sum(1, keepdim=True)  # normalise rows

    # COO edge index: fully connected
    src, dst = torch.meshgrid(torch.arange(N_NODES), torch.arange(N_NODES), indexing="ij")
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight, adj


def test_model_output_shape(graph_tensors):
    import torch
    from scripts.models.st_gnn import STGNN

    model = STGNN(n_features=N_FEATURES, n_nodes=N_NODES,
                  spatial_hidden=16, spatial_out=16, n_heads=2,
                  lstm_hidden=16, lstm_layers=1,
                  decoder_hidden=16)
    model.eval()

    x = torch.randn(B, T, N_NODES, N_FEATURES)
    edge_index, edge_weight, adj = graph_tensors

    with torch.no_grad():
        y = model(x, edge_index, edge_weight, adj)

    assert y.shape == (B, N_NODES, 1), f"Expected ({B}, {N_NODES}, 1), got {y.shape}"


def test_model_output_range(graph_tensors):
    import torch
    from scripts.models.st_gnn import STGNN

    model = STGNN(n_features=N_FEATURES, n_nodes=N_NODES,
                  spatial_hidden=16, spatial_out=16, n_heads=2,
                  lstm_hidden=16, lstm_layers=1, decoder_hidden=16)
    model.eval()
    x = torch.randn(2, T, N_NODES, N_FEATURES)
    edge_index, edge_weight, adj = graph_tensors

    with torch.no_grad():
        y = model(x, edge_index, edge_weight, adj)

    assert (y >= 0).all() and (y <= 1).all(), "S4 predictions must be in [0, 1]"


def test_combined_loss():
    import torch
    from scripts.models.losses import CombinedLoss

    loss_fn = CombinedLoss(mse_weight=0.7, mae_weight=0.3)
    y_pred = torch.tensor([0.3, 0.5, 0.7])
    y_true = torch.tensor([0.2, 0.6, 0.8])
    loss = loss_fn(y_pred, y_true)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_focal_loss_severe_upweighting():
    import torch
    from scripts.models.losses import FocalScintillationLoss

    loss_fn = FocalScintillationLoss(severe_threshold=0.7, severe_weight=3.0)
    # Severe event: large error should give large loss
    y_pred_bad = torch.tensor([0.1, 0.1])
    y_true_severe = torch.tensor([0.8, 0.9])
    # Quiet: same error magnitude
    y_pred_ok = torch.tensor([0.1, 0.1])
    y_true_quiet = torch.tensor([0.2, 0.3])

    loss_severe = loss_fn(y_pred_bad, y_true_severe)
    loss_quiet = loss_fn(y_pred_ok, y_true_quiet)
    assert loss_severe > loss_quiet, "Severe events should have higher loss"


def test_regression_metrics():
    from scripts.models.metrics import compute_regression_metrics
    y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_pred = np.array([0.15, 0.25, 0.55, 0.65, 0.95])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]  # RMSE ≥ MAE always


def test_event_metrics_perfect():
    from scripts.models.metrics import compute_event_metrics
    y_true = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
    y_pred = y_true.copy()
    metrics = compute_event_metrics(y_true, y_pred, {"warning": 0.4})
    assert abs(metrics["warning"]["f1"] - 1.0) < 1e-6, "Perfect forecast should have F1=1"


def test_mc_dropout_uncertainty(graph_tensors):
    import torch
    from scripts.models.st_gnn import STGNN

    model = STGNN(n_features=N_FEATURES, n_nodes=N_NODES,
                  spatial_hidden=16, spatial_out=16, n_heads=2,
                  lstm_hidden=16, lstm_layers=1, decoder_hidden=16)

    x = torch.randn(1, T, N_NODES, N_FEATURES)
    edge_index, edge_weight, adj = graph_tensors

    mean, std = model.predict_with_dropout(x, edge_index, edge_weight, adj, n_samples=5)
    assert mean.shape == (1, N_NODES, 1)
    assert std.shape == (1, N_NODES, 1)
    assert (std >= 0).all()
