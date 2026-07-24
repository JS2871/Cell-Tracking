"""Frame-to-frame linking of per-timepoint detections into a tracked graph.

Builds a `tracksdata` graph from detections, adds candidate edges between
consecutive timepoints within the competition's 7um matching radius (via
`tracksdata.edges.DistanceEdges`, using each node's *physical* micron
coordinates so anisotropic Z spacing is handled correctly), then solves it
with `tracksdata.solvers.NearestNeighborsSolver`. `max_children=2` on the
solver *is* the division model: a node keeps at most two children (one
parent per node is enforced too), so no separate division heuristic is
needed on top of it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.constants import DEFAULT_ATTR_KEYS as K
from tracksdata.edges import DistanceEdges
from tracksdata.solvers import NearestNeighborsSolver

PHYS_KEYS = ("z_phys", "y_phys", "x_phys")


def build_candidate_graph(
    detections: dict[int, np.ndarray],
    scale: tuple[float, float, float],
) -> td.graph.RustWorkXGraph:
    """Build a graph with one node per detection (no edges yet).

    Parameters
    ----------
    detections : dict[int, np.ndarray]
        Maps timepoint -> (N, 3) array of (z, y, x) voxel coordinates, as
        returned by :func:`src.classical.detect.detect_volume`.
    scale : tuple[float, float, float]
        Physical (Z, Y, X) voxel scale, in microns.

    Returns
    -------
    tracksdata.graph.RustWorkXGraph
    """
    graph = td.graph.RustWorkXGraph()
    for key in (K.Z, K.Y, K.X, *PHYS_KEYS):
        graph.add_node_attr_key(key, pl.Float64)

    scale_arr = np.array(scale)
    for t, coords in detections.items():
        if len(coords) == 0:
            continue
        phys = coords * scale_arr
        nodes = [
            {
                K.T: t,
                K.Z: float(z), K.Y: float(y), K.X: float(x),
                "z_phys": float(pz), "y_phys": float(py), "x_phys": float(px),
            }
            for (z, y, x), (pz, py, px) in zip(coords, phys)
        ]
        graph.bulk_add_nodes(nodes)

    return graph


def link_detections(
    detections: dict[int, np.ndarray],
    scale: tuple[float, float, float],
    max_distance_um: float = 7.0,
    n_neighbors: int = 5,
    max_children: int = 2,
) -> td.graph.BaseGraph:
    """Detect -> candidate edges -> nearest-neighbor solve, in one call.

    Parameters
    ----------
    detections : dict[int, np.ndarray]
        Per-timepoint (z, y, x) voxel-coordinate detections.
    scale : tuple[float, float, float]
        Physical (Z, Y, X) voxel scale, in microns.
    max_distance_um : float
        Matching radius in microns (7.0 matches the competition metric's own
        node-matching threshold).
    n_neighbors : int
        Candidate edges kept per node before solving (must be >= max_children
        for divisions to be representable).
    max_children : int
        Max outgoing edges per node in the solved graph (2 allows divisions).

    Returns
    -------
    tracksdata.graph.BaseGraph
        The solved graph (nodes/edges restricted to the accepted solution).
    """
    graph = build_candidate_graph(detections, scale)

    if graph.num_nodes() == 0:
        return graph

    edge_op = DistanceEdges(
        distance_threshold=max_distance_um,
        n_neighbors=n_neighbors,
        attr_keys=list(PHYS_KEYS),
    )
    edge_op.add_edges(graph)

    if graph.num_edges() == 0:
        return graph

    solver = NearestNeighborsSolver(max_children=max_children, edge_weight=K.EDGE_DIST)
    return solver.solve(graph)
