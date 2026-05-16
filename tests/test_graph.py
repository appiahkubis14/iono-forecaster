"""
tests/test_graph.py — IonoForecaster
Tests for graph construction.

Author: Samuel Appiah Kubi
"""

import numpy as np
import pytest


STATIONS = [
    {"id": "ACCR", "lat": 5.56, "lon": -0.20},
    {"id": "LAGO", "lat": 6.52, "lon":  3.38},
    {"id": "NKLG", "lat": 0.35, "lon":  9.67},
    {"id": "MBAR", "lat": -0.60, "lon": 30.74},
    {"id": "NAIR", "lat": -1.22, "lon": 36.89},
]


def test_graph_build_shape():
    from scripts.graph.build_graph import StationGraph
    g = StationGraph(STATIONS, distance_threshold_km=5000)
    g.build()
    N = len(STATIONS)
    assert g.adj_matrix.shape == (N, N)


def test_graph_self_loops():
    from scripts.graph.build_graph import StationGraph
    g = StationGraph(STATIONS, self_loops=True)
    g.build()
    diag = np.diag(g.adj_matrix)
    assert (diag == 1.0).all(), "Self-loops should be 1.0"


def test_graph_edge_index_valid():
    from scripts.graph.build_graph import StationGraph
    g = StationGraph(STATIONS, distance_threshold_km=5000)
    g.build()
    ei = g.edge_index
    N = len(STATIONS)
    assert ei.shape[0] == 2
    assert ei.max() < N
    assert ei.min() >= 0


def test_graph_save_load(tmp_path):
    from scripts.graph.build_graph import StationGraph
    g = StationGraph(STATIONS)
    g.build()
    g.save(tmp_path)

    g2 = StationGraph.load(tmp_path)
    np.testing.assert_array_almost_equal(g.adj_matrix, g2.adj_matrix)


def test_haversine_known():
    from scripts.utils import haversine_km
    # Accra to Lagos: approx 450 km
    d = haversine_km(5.56, -0.20, 6.52, 3.38)
    assert 400 < d < 500, f"Unexpected distance: {d:.1f} km"
