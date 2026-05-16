"""
models/st_gnn.py — IonoForecaster
Spatio-Temporal Graph Neural Network (ST-GNN) for ionospheric scintillation forecasting.

Architecture:
  1. Spatial encoder: GATv2 (Graph Attention Network v2) — multi-head attention
  2. Temporal encoder: 2-layer Bidirectional LSTM
  3. Decoder: Fully-connected with residual connection
  4. Output: S4 index at each node for next 30-minute timestep

References:
  - Brody et al. (2022) "How Attentive are Graph Attention Networks?" (GATv2)
  - Wu et al. (2019) "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports — optional, gracefully degraded if not installed
try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ─────────────────────────────────────────────────────────────────────────────
# Fallback GAT for environments without PyG
# ─────────────────────────────────────────────────────────────────────────────

class SimpleGATLayer(nn.Module):
    """
    Simplified Graph Attention layer (GATv1 style) using sparse matrix ops.
    Fallback when torch_geometric is not available.

    Parameters
    ----------
    in_features : int
    out_features : int
    num_heads : int
    dropout : float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        # Per-head linear transformations
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        # Attention vectors
        self.a_src = nn.Parameter(torch.zeros(1, num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(1, num_heads, out_features))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_features)
        adj: torch.Tensor,         # (N, N) adjacency (weighted)
    ) -> torch.Tensor:
        N = x.size(0)
        # Linear projection → (N, H, out_features)
        Wx = self.W(x).view(N, self.num_heads, self.out_features)

        # Attention scores: e_ij = LeakyReLU(a_src·Wx_i + a_dst·Wx_j)
        e_src = (Wx * self.a_src).sum(-1, keepdim=True)  # (N, H, 1)
        e_dst = (Wx * self.a_dst).sum(-1, keepdim=True)  # (N, H, 1)
        # Broadcast
        e = self.leaky_relu(e_src + e_dst.transpose(0, 2).unsqueeze(0))  # (N, N, H, 1)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).unsqueeze(-1)
        e = e.masked_fill(mask, float("-inf"))

        # Softmax over neighbours
        alpha = F.softmax(e, dim=1)  # (N, N, H, 1)
        alpha = self.dropout(alpha)

        # Aggregate
        out = (alpha * Wx.unsqueeze(0)).sum(1)  # (N, H, out_features)
        if self.concat:
            return out.reshape(N, self.num_heads * self.out_features)
        return out.mean(1)  # (N, out_features)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial encoder
# ─────────────────────────────────────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    GATv2 spatial encoder.

    Applies graph attention over the station graph to capture spatial
    correlations of ionospheric irregularities.

    Parameters
    ----------
    in_features : int
        Number of input node features.
    hidden_dim : int
    out_dim : int
    num_heads : int
    dropout : float
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if HAS_PYG:
            self.gat1 = GATv2Conv(
                in_features, hidden_dim, heads=num_heads, dropout=dropout, concat=True
            )
            self.gat2 = GATv2Conv(
                hidden_dim * num_heads, out_dim, heads=1, dropout=dropout, concat=False
            )
        else:
            self.gat1 = SimpleGATLayer(in_features, hidden_dim, num_heads, dropout, concat=True)
            self.gat2 = SimpleGATLayer(hidden_dim * num_heads, out_dim, 1, dropout, concat=False)

        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.use_pyg = HAS_PYG

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (N, in_features) node feature matrix.
        edge_index : torch.Tensor
            (2, E) edge indices (for PyG) or None.
        edge_weight : torch.Tensor
            (E,) edge weights (for PyG).
        adj : torch.Tensor
            (N, N) adjacency matrix (for fallback).

        Returns
        -------
        torch.Tensor
            (N, out_dim) node embeddings.
        """
        if self.use_pyg:
            h = F.elu(self.gat1(x, edge_index, edge_weight))
            h = self.norm1(h)
            h = self.dropout(h)
            h = self.norm2(self.gat2(h, edge_index, edge_weight))
        else:
            assert adj is not None, "adj required for fallback GAT"
            h = F.elu(self.gat1(x, adj))
            h = self.norm1(h)
            h = self.dropout(h)
            h = self.norm2(self.gat2(h, adj))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Temporal encoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Bidirectional LSTM temporal encoder.

    Processes the sequence of spatial embeddings over T timesteps.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    num_layers : int
    dropout : float
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_size * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, T, input_size) sequence of spatial embeddings.

        Returns
        -------
        torch.Tensor
            (B, hidden_size * 2) final hidden state.
        """
        out, (h_n, _) = self.lstm(x)
        # Concatenate forward and backward last hidden states
        h_fwd = h_n[-2]   # forward last layer
        h_bwd = h_n[-1]   # backward last layer
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.dropout(h)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Fully-connected decoder with residual connection.

    Parameters
    ----------
    in_dim : int
    hidden_dim : int
    out_dim : int
    dropout : float
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, in_dim) or (N, in_dim).

        Returns
        -------
        torch.Tensor
            (B, out_dim) or (N, out_dim) — S4 prediction.
        """
        residual = self.residual(x)
        h = F.relu(self.norm(self.fc1(x)))
        h = self.dropout(h)
        return self.fc2(h) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Full ST-GNN
# ─────────────────────────────────────────────────────────────────────────────

class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for ionospheric scintillation forecasting.

    Input tensor shape: (B, T, N, F)
      B = batch size
      T = input timesteps (12 = 6 hours at 30-min resolution)
      N = number of GNSS stations (nodes)
      F = features per node

    Output shape: (B, N, 1)
      S4 index prediction for next 30-minute window.

    Parameters
    ----------
    n_features : int
        Number of input features per node.
    n_nodes : int
        Number of stations.
    spatial_hidden : int
    spatial_out : int
    n_heads : int
    lstm_hidden : int
    lstm_layers : int
    decoder_hidden : int
    dropout_spatial : float
    dropout_temporal : float
    dropout_decoder : float
    """

    def __init__(
        self,
        n_features: int,
        n_nodes: int,
        spatial_hidden: int = 64,
        spatial_out: int = 64,
        n_heads: int = 4,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        decoder_hidden: int = 128,
        dropout_spatial: float = 0.2,
        dropout_temporal: float = 0.3,
        dropout_decoder: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.spatial_out = spatial_out

        # Input projection
        self.input_proj = nn.Linear(n_features, spatial_hidden)

        # Spatial encoder (shared across timesteps)
        self.spatial = SpatialEncoder(
            in_features=spatial_hidden,
            hidden_dim=spatial_hidden,
            out_dim=spatial_out,
            num_heads=n_heads,
            dropout=dropout_spatial,
        )

        # Temporal encoder (one per node, weight-shared)
        self.temporal = TemporalEncoder(
            input_size=spatial_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout_temporal,
        )

        # Decoder
        self.decoder = Decoder(
            in_dim=lstm_hidden * 2,
            hidden_dim=decoder_hidden,
            out_dim=1,
            dropout=dropout_decoder,
        )

        # Sigmoid to keep S4 in [0, 1]
        self.output_activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            (B, T, N, F) input tensor.
        edge_index : torch.Tensor
            (2, E) PyG edge index.
        edge_weight : torch.Tensor, optional
            (E,) edge weights.
        adj : torch.Tensor, optional
            (N, N) dense adjacency (for fallback).

        Returns
        -------
        torch.Tensor
            (B, N, 1) S4 predictions.
        """
        B, T, N, F = x.shape
        assert N == self.n_nodes

        # Project input features for all timesteps and nodes
        x_proj = self.input_proj(x.reshape(B * T * N, F)).reshape(B, T, N, -1)

        # Apply spatial encoder at each timestep
        spatial_out_list = []
        for t in range(T):
            x_t = x_proj[:, t, :, :]  # (B, N, spatial_hidden)
            # Process each batch item through GAT
            if B == 1:
                node_emb = self.spatial(x_t[0], edge_index, edge_weight, adj)
                node_emb = node_emb.unsqueeze(0)
            else:
                # Process batch sequentially (GAT operates per-graph)
                node_embs = []
                for b in range(B):
                    emb = self.spatial(x_t[b], edge_index, edge_weight, adj)
                    node_embs.append(emb)
                node_emb = torch.stack(node_embs, dim=0)
            spatial_out_list.append(node_emb)  # (B, N, spatial_out)

        # Stack over time: (B, T, N, spatial_out)
        spatial_seq = torch.stack(spatial_out_list, dim=1)

        # Temporal encoding: process each node's time series
        # Reshape to (B*N, T, spatial_out)
        spatial_seq_flat = spatial_seq.transpose(1, 2).reshape(B * N, T, self.spatial_out)
        temporal_out = self.temporal(spatial_seq_flat)  # (B*N, lstm_hidden*2)

        # Decode
        pred = self.decoder(temporal_out)  # (B*N, 1)
        pred = pred.reshape(B, N, 1)
        pred = self.output_activation(pred)
        return pred

    def predict_with_dropout(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor], adj: Optional[torch.Tensor],
        n_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference for uncertainty quantification.

        Keeps dropout active during inference and runs n_samples forward passes.

        Returns
        -------
        mean_pred : torch.Tensor (B, N, 1)
        std_pred : torch.Tensor (B, N, 1)
        """
        self.train()  # activate dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, edge_index, edge_weight, adj)
                preds.append(pred)
        self.eval()

        preds_stacked = torch.stack(preds, dim=0)  # (n_samples, B, N, 1)
        mean_pred = preds_stacked.mean(0)
        std_pred = preds_stacked.std(0)
        return mean_pred, std_pred


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_features: int, n_nodes: int) -> STGNN:
    """
    Instantiate STGNN from config.yaml model section.

    Parameters
    ----------
    cfg : dict
    n_features : int
    n_nodes : int

    Returns
    -------
    STGNN
    """
    m = cfg.get("model", {})
    spatial = m.get("spatial", {})
    temporal = m.get("temporal", {})
    decoder = m.get("decoder", {})

    model = STGNN(
        n_features=n_features,
        n_nodes=n_nodes,
        spatial_hidden=spatial.get("hidden_dim", 64),
        spatial_out=spatial.get("output_dim", 64),
        n_heads=spatial.get("num_heads", 4),
        lstm_hidden=temporal.get("hidden_size", 128),
        lstm_layers=temporal.get("num_layers", 2),
        decoder_hidden=decoder.get("hidden_dim", 128),
        dropout_spatial=spatial.get("dropout", 0.2),
        dropout_temporal=temporal.get("dropout", 0.3),
        dropout_decoder=0.3,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    from scripts.utils import get_logger
    get_logger("model").info(
        "ST-GNN built: %d trainable parameters | %d nodes | %d features",
        n_params, n_nodes, n_features,
    )
    return model
