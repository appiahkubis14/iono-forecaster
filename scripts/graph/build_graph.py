"""
graph/build_graph.py — IonoForecaster
Build the spatial adjacency graph for ST-GNN.

Graph construction rules:
  1. Geographic distance threshold (default 2000 km)
  2. TEC correlation threshold (default 0.7)
  3. Edge weights = inverse distance (normalised)
  4. Self-loops included

The graph is stored as:
  - adjacency matrix (N × N NumPy array)
  - edge_index (2 × E tensor) for PyG
  - edge_weight (E,) tensor

Author: Samuel Appiah Kubi
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.utils import get_logger, haversine_km

logger = get_logger("graph.build_graph")


class StationGraph:
    """
    Build and manage the spatial station graph for the ST-GNN.

    Parameters
    ----------
    stations : list of dict
        Each dict must have: id, lat, lon.
    distance_threshold_km : float
    correlation_threshold : float
    self_loops : bool
    """

    def __init__(
        self,
        stations: List[Dict],
        distance_threshold_km: float = 2000.0,
        correlation_threshold: float = 0.7,
        self_loops: bool = True,
    ) -> None:
        self.stations = stations
        self.n = len(stations)
        self.distance_threshold_km = distance_threshold_km
        self.correlation_threshold = correlation_threshold
        self.self_loops = self_loops

        self.station_ids = [s["id"] for s in stations]
        self.lats = np.array([s["lat"] for s in stations])
        self.lons = np.array([s["lon"] for s in stations])

        self._dist_matrix: Optional[np.ndarray] = None
        self._adj_matrix: Optional[np.ndarray] = None
        self._edge_index: Optional[np.ndarray] = None
        self._edge_weight: Optional[np.ndarray] = None

    # ── public ────────────────────────────────────────────────────────────────

    def build(
        self,
        tec_data: Optional[pd.DataFrame] = None,
    ) -> "StationGraph":
        """
        Construct the graph adjacency matrix.

        Parameters
        ----------
        tec_data : pd.DataFrame, optional
            Long-format TEC data (columns: timestamp, station_id, tec) used
            to compute temporal correlation edges. If None, only distance
            edges are used.

        Returns
        -------
        self
        """
        logger.info(
            "Building station graph: %d nodes, dist_thresh=%.0f km, corr_thresh=%.2f",
            self.n, self.distance_threshold_km, self.correlation_threshold,
        )

        self._build_distance_matrix()
        adj = self._distance_edges()

        if tec_data is not None:
            corr_adj = self._correlation_edges(tec_data)
            # Union: edge exists if distance OR correlation criterion met
            adj = np.logical_or(adj, corr_adj).astype(np.float32)
        else:
            adj = adj.astype(np.float32)

        # Weight by inverse distance
        dist_safe = np.where(self._dist_matrix > 0, self._dist_matrix, np.inf)
        weights = 1.0 / dist_safe
        weights = weights / weights[np.isfinite(weights)].max()  # normalise
        adj_weighted = adj * weights

        if self.self_loops:
            np.fill_diagonal(adj_weighted, 1.0)

        self._adj_matrix = adj_weighted
        self._edge_index, self._edge_weight = self._adj_to_edge_index(adj_weighted)

        n_edges = self._edge_index.shape[1]
        logger.info("Graph built: %d nodes, %d edges.", self.n, n_edges)
        return self

    def save(self, out_dir: str | Path) -> None:
        """Save graph artefacts to disk."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "adj_matrix.npy", self._adj_matrix)
        np.save(out_dir / "edge_index.npy", self._edge_index)
        np.save(out_dir / "edge_weight.npy", self._edge_weight)

        meta = {
            "station_ids": self.station_ids,
            "lats": self.lats.tolist(),
            "lons": self.lons.tolist(),
            "distance_threshold_km": self.distance_threshold_km,
            "correlation_threshold": self.correlation_threshold,
            "n_nodes": self.n,
            "n_edges": int(self._edge_index.shape[1]),
        }
        with open(out_dir / "graph_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Graph saved → %s", out_dir)

    @classmethod
    def load(cls, graph_dir: str | Path) -> "StationGraph":
        """Load a previously saved graph."""
        graph_dir = Path(graph_dir)
        with open(graph_dir / "graph_meta.json") as fh:
            meta = json.load(fh)

        stations = [
            {"id": sid, "lat": lat, "lon": lon}
            for sid, lat, lon in zip(meta["station_ids"], meta["lats"], meta["lons"])
        ]
        obj = cls(
            stations,
            distance_threshold_km=meta["distance_threshold_km"],
            correlation_threshold=meta["correlation_threshold"],
        )
        obj._adj_matrix = np.load(graph_dir / "adj_matrix.npy")
        obj._edge_index = np.load(graph_dir / "edge_index.npy")
        obj._edge_weight = np.load(graph_dir / "edge_weight.npy")
        logger.info(
            "Graph loaded: %d nodes, %d edges.", meta["n_nodes"], meta["n_edges"]
        )
        return obj

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def adj_matrix(self) -> np.ndarray:
        if self._adj_matrix is None:
            raise RuntimeError("Graph not built yet. Call .build() first.")
        return self._adj_matrix

    @property
    def edge_index(self) -> np.ndarray:
        """(2, E) array of [source, target] edge indices."""
        if self._edge_index is None:
            raise RuntimeError("Graph not built yet.")
        return self._edge_index

    @property
    def edge_weight(self) -> np.ndarray:
        """(E,) array of edge weights."""
        if self._edge_weight is None:
            raise RuntimeError("Graph not built yet.")
        return self._edge_weight

    def to_pyg(self) -> Tuple:
        """
        Return edge_index and edge_weight as PyTorch tensors for PyG.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            edge_index (2, E, dtype=long), edge_weight (E, dtype=float)
        """
        import torch
        ei = torch.from_numpy(self._edge_index).long()
        ew = torch.from_numpy(self._edge_weight).float()
        return ei, ew

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_distance_matrix(self) -> None:
        dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dist[i, j] = haversine_km(
                        self.lats[i], self.lons[i],
                        self.lats[j], self.lons[j],
                    )
        self._dist_matrix = dist

    def _distance_edges(self) -> np.ndarray:
        """Adjacency from distance threshold."""
        adj = (self._dist_matrix <= self.distance_threshold_km).astype(float)
        if not self.self_loops:
            np.fill_diagonal(adj, 0)
        return adj

    def _correlation_edges(self, tec_data: pd.DataFrame) -> np.ndarray:
        """
        Adjacency from temporal TEC correlation between stations.

        Expects columns: timestamp, station_id, tec.
        """
        adj = np.zeros((self.n, self.n))
        try:
            pivot = tec_data.pivot_table(
                index="timestamp", columns="station_id", values="tec"
            )
            # Keep only known stations
            pivot = pivot.reindex(columns=self.station_ids)
            corr_matrix = pivot.corr().values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            for i in range(self.n):
                for j in range(self.n):
                    if i != j and abs(corr_matrix[i, j]) >= self.correlation_threshold:
                        adj[i, j] = 1.0
        except Exception as exc:
            logger.warning("Correlation edge computation failed: %s", exc)
        return adj

    @staticmethod
    def _adj_to_edge_index(
        adj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert adjacency matrix to COO edge_index format."""
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
        edge_weight = adj[rows, cols].astype(np.float32)
        return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(cfg: Dict, gnss_data: Optional[pd.DataFrame] = None) -> StationGraph:
    """
    Build the station graph and save to disk.

    Parameters
    ----------
    cfg : dict
    gnss_data : pd.DataFrame, optional
        Merged GNSS TEC data for correlation-based edges.

    Returns
    -------
    StationGraph
    """
    graph_cfg = cfg.get("graph", {})
    stations = cfg.get("stations", [])

    if not stations:
        from scripts.downloaders.gnss_samba import STATION_META
        stations = STATION_META

    graph = StationGraph(
        stations=stations,
        distance_threshold_km=graph_cfg.get("distance_threshold_km", 2000),
        correlation_threshold=graph_cfg.get("correlation_threshold", 0.7),
        self_loops=graph_cfg.get("self_loops", True),
    )
    graph.build(tec_data=gnss_data)

    out_dir = Path(cfg["paths"]["graphs_data"])
    graph.save(out_dir)
    return graph
