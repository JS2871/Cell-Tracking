"""3D per-timepoint cell detection for the classical baseline.

Smooths each timepoint's volume, thresholds it against the sample's own
intensity quantiles (from the zarr ``image_statistics`` metadata), and finds
local maxima as candidate cell centers. Peaks are then thinned with the
vendored ``tracking_cellmot.img_proc.nms_3d`` so nearby duplicate peaks
(in physical microns) collapse to one detection per cell.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from tracking_cellmot.img_proc import nms_3d


def detect_frame(
    volume: np.ndarray,
    scale: tuple[float, float, float],
    intensity_threshold: float,
    sigma: float = 1.0,
    min_distance_um: float = 4.0,
) -> np.ndarray:
    """Detect candidate cell centers in one (Z, Y, X) volume.

    Parameters
    ----------
    volume : np.ndarray
        Single-timepoint (Z, Y, X) volume.
    scale : tuple[float, float, float]
        Physical (Z, Y, X) voxel scale, in microns.
    intensity_threshold : float
        Minimum smoothed intensity for a voxel to be considered a peak
        (typically a sample's own high intensity quantile, e.g. the 0.9 or
        0.99 quantile from the zarr ``image_statistics`` metadata).
    sigma : float
        Gaussian smoothing sigma (in voxels) applied before peak detection.
    min_distance_um : float
        Minimum physical distance (microns) between kept peaks.

    Returns
    -------
    np.ndarray
        (N, 3) array of (z, y, x) integer voxel coordinates.
    """
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma)
    coords = peak_local_max(smoothed, min_distance=1, threshold_abs=intensity_threshold)
    if len(coords) == 0:
        return coords.astype(np.int64)

    scores = smoothed[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = nms_3d(coords, scores, min_distance=min_distance_um, scale=scale)
    return coords[keep].astype(np.int64)


def detect_volume(
    image: np.ndarray,
    scale: tuple[float, float, float],
    quantiles: dict[str, float],
    threshold_quantile: str = "0.9",
    sigma: float = 1.0,
    min_distance_um: float = 4.0,
) -> dict[int, np.ndarray]:
    """Detect candidate cell centers at every timepoint of a (T, Z, Y, X) volume.

    Parameters
    ----------
    image : np.ndarray
        Full (T, Z, Y, X) volume.
    scale : tuple[float, float, float]
        Physical (Z, Y, X) voxel scale, in microns.
    quantiles : dict[str, float]
        The sample's ``image_statistics.quantiles`` dict (from zarr attrs),
        e.g. ``{"0.9": 497.0, "0.99": 1478.0, ...}``.
    threshold_quantile : str
        Which key of ``quantiles`` to use as the detection intensity threshold.
    sigma : float
        Gaussian smoothing sigma (in voxels).
    min_distance_um : float
        Minimum physical distance (microns) between kept peaks.

    Returns
    -------
    dict[int, np.ndarray]
        Maps timepoint index -> (N, 3) array of (z, y, x) voxel coordinates.
    """
    threshold = quantiles[threshold_quantile]
    detections = {}
    for t in range(image.shape[0]):
        detections[t] = detect_frame(
            image[t], scale, threshold, sigma=sigma, min_distance_um=min_distance_um,
        )
    return detections
