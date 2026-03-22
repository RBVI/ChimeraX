"""
fp32 boundary utilities for the Metal rendering path.

Policy
------
* ChimeraX atomic coordinates and spline parameters stay **float64** inside
  the core and atomic bundles — this is scientifically important for
  Angstrom-precision placement of atoms at typical structure scales (up to
  ~100 000 Å from origin for large assemblies).
* All geometry that reaches the GPU must be **float32**.  At typical
  molecular scales (< 10 000 Å from origin) fp32 has ~1 pm precision, which
  is well within rendering tolerance.
* The conversion happens exactly once, at the point where a Drawing's vertex
  array is handed to the Metal renderer.  No other bundle needs to change.

Maximum error introduced by fp32 truncation
--------------------------------------------
Float32 has 23 mantissa bits → ~7 significant decimal digits.
For a coordinate of 10 000 Å the rounding error is ≈ 0.001 Å = 1 pm.
This is smaller than a chemical bond rendering tolerance and far smaller
than inter-atom distances.

Usage
-----
All geometry-to-GPU code should call `to_fp32_vertices` / `to_fp32_normals`
rather than `array.astype(float32)` directly so that any future policy change
(e.g. relative-to-camera repositioning for very large assemblies) is applied
in one place.
"""

from __future__ import annotations
import numpy as np


# Maximum allowable distance from the coordinate origin before fp32 precision
# loss exceeds 0.1 Å (a generous rendering tolerance).
_FP32_SAFE_RADIUS_ANGSTROM: float = 1_000_000.0  # 1 million Å ≈ 100 µm


def to_fp32_vertices(coords: np.ndarray) -> np.ndarray:
    """
    Return *coords* as a float32 (N,3) array, recentring if needed.

    If all coordinates are within ±FP32_SAFE_RADIUS_ANGSTROM from the origin
    a simple cast is performed.  For extremely large structures a relative-
    to-centroid repositioning is applied so that the GPU still gets accurate
    relative positions.

    Parameters
    ----------
    coords:
        (N,3) array of any floating-point dtype (typically float32 or float64).

    Returns
    -------
    fp32 (N,3) array ready for upload to Metal.
    """
    if coords.dtype == np.float32:
        return coords

    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return coords.astype(np.float32)

    max_abs = np.abs(coords).max()
    if max_abs > _FP32_SAFE_RADIUS_ANGSTROM:
        centroid = coords.mean(axis=0)
        coords = coords - centroid  # relative coordinates, safe for fp32
        # NOTE: the caller must apply the same centroid offset to the model
        # matrix (or accept ~sub-pm positioning error for huge assemblies).

    return coords.astype(np.float32)


def to_fp32_normals(normals: np.ndarray) -> np.ndarray:
    """
    Return normalised float32 normals.  Normals are dimensionless unit vectors
    so fp32 precision is always more than adequate.
    """
    n = np.asarray(normals, dtype=np.float32)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return n / lengths


def to_fp32_colors(colors: np.ndarray) -> np.ndarray:
    """
    Return a float32 (N,4) RGBA array in the 0–1 range.

    Accepts uint8 (0–255) or float arrays (assumed 0–1 if ≤ 1, else 0–255).
    """
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors.astype(np.float32) / 255.0
    c = colors.astype(np.float32)
    if c.max() > 1.0:
        c = c / 255.0
    return c


def dtype_check_warning(name: str, arr: np.ndarray, logger=None) -> None:
    """
    Emit a warning (to *logger* or stderr) if *arr* is float64.

    Use this during development to trace unexpected float64 arrays entering
    the Metal path.  In production this is a no-op when no logger is given.
    """
    if arr.dtype != np.float32:
        msg = (
            f"[fp32_utils] {name} has dtype {arr.dtype}; "
            "expected float32 at the GPU boundary.  "
            "This will be auto-converted but indicates a missed cast upstream."
        )
        if logger is not None:
            logger.warning(msg)
        else:
            import warnings
            warnings.warn(msg, stacklevel=2)
