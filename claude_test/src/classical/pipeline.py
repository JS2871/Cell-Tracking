"""End-to-end classical baseline for one sample: image -> detections -> tracked graph -> .geff.

Reads directly from an extracted `<id>.zarr` directory (not from the zip -
the vendored `tracking_cellmot`/`tracksdata` tooling expects plain
directories), detects candidate cells per timepoint, links them into a
tracked graph, and saves the result as a `.geff` via
`tracking_cellmot.io.save_graph`.
"""

from __future__ import annotations

from pathlib import Path

import zarr

from tracking_cellmot.io import save_graph

from .detect import detect_volume
from .track import link_detections


def _open_image(zarr_path: Path) -> tuple:
    g = zarr.open_group(zarr_path, mode="r")
    arr = g["0"]
    image = arr[:]
    attrs = dict(g.attrs)
    scale = tuple(attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][-3:])
    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})
    return image, scale, quantiles


def run_sample(
    zarr_path: Path | str,
    output_geff_path: Path | str,
    threshold_quantile: str = "0.9",
    sigma: float = 1.0,
    min_distance_um: float = 4.0,
    max_distance_um: float = 7.0,
) -> dict:
    """Run the classical baseline on one sample and save the predicted `.geff`.

    Parameters
    ----------
    zarr_path : Path or str
        Path to the sample's extracted `<id>.zarr` directory.
    output_geff_path : Path or str
        Where to write the predicted `.geff`.
    threshold_quantile, sigma, min_distance_um, max_distance_um
        Forwarded to :func:`src.classical.detect.detect_volume` /
        :func:`src.classical.track.link_detections`.

    Returns
    -------
    dict
        Summary: number of detections per timepoint (total), nodes, edges.
    """
    zarr_path = Path(zarr_path)
    image, scale, quantiles = _open_image(zarr_path)

    detections = detect_volume(
        image, scale, quantiles,
        threshold_quantile=threshold_quantile, sigma=sigma, min_distance_um=min_distance_um,
    )
    n_detections = sum(len(v) for v in detections.values())

    graph = link_detections(detections, scale, max_distance_um=max_distance_um)
    save_graph(graph, output_geff_path)

    return {
        "sample_id": zarr_path.stem,
        "n_detections": n_detections,
        "n_nodes": graph.num_nodes(),
        "n_edges": graph.num_edges(),
    }
