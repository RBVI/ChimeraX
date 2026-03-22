"""
Drawing-to-Metal geometry uploader.

walk_and_upload() traverses a chimerax.graphics.Drawing tree and uploads
fp32 geometry into Metal buffers that MetalRenderer can encode.

Pass ordering
-------------
1. Opaque solid triangles (depth write on, front-to-back preferred)
2. Transparent triangles (depth write off, back-to-front)

Each geometry batch is a dict with keys that map to MetalRenderer methods:
  - 'vertices'   float32 (N,3)
  - 'normals'    float32 (N,3)
  - 'colors'     float32 (N,4)   RGBA 0-1
  - 'indices'    int32   (M,3)
  - 'transparent' bool
  - 'positions'  list of 4x4 float32 numpy arrays (one per copy)
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def walk_and_upload(drawing, renderer, context) -> None:
    """
    Walk *drawing* and its children, uploading geometry to *renderer*.

    Parameters
    ----------
    drawing:
        Root chimerax.graphics.Drawing.
    renderer:
        A PyMetalRenderer Cython wrapper.
    context:
        A PyMetalContext Cython wrapper (used for buffer creation).
    """
    opaque_batches = []
    transparent_batches = []

    _collect(drawing, opaque_batches, transparent_batches)

    # Upload opaque first, then transparent.
    for batch in opaque_batches:
        _upload_batch(batch, renderer, context, transparent=False)
    for batch in transparent_batches:
        _upload_batch(batch, renderer, context, transparent=True)


# --------------------------------------------------------------------------
# Tree traversal
# --------------------------------------------------------------------------

def _collect(drawing, opaque: list, transparent: list) -> None:
    """Recursively collect geometry batches from *drawing*."""
    if not drawing.display:
        return

    vertices = drawing.vertices
    triangles = drawing.triangles

    if vertices is not None and triangles is not None and len(triangles) > 0:
        # Ensure fp32 — Drawing spec says float32, but guard anyway.
        verts = _as_fp32(vertices)
        from .fp32_utils import to_fp32_normals
        normals = to_fp32_normals(drawing.normals) if drawing.normals is not None \
            else _compute_normals(verts, triangles)
        colors = _resolve_colors(drawing, verts)
        inds = triangles.astype(np.int32, copy=False)

        # Expand per-position copies into individual batches.
        positions = _get_positions(drawing)
        for pos_mat in positions:
            # Transform vertices and normals into world space for this copy.
            w_verts = _transform_points(verts, pos_mat)
            w_norms = _transform_normals(normals, pos_mat)
            batch = {
                "vertices": w_verts,
                "normals": w_norms,
                "colors": colors,
                "indices": inds,
            }
            if _is_transparent(drawing):
                transparent.append(batch)
            else:
                opaque.append(batch)

    for child in drawing.child_drawings():
        _collect(child, opaque, transparent)


# --------------------------------------------------------------------------
# Buffer upload
# --------------------------------------------------------------------------

def _upload_batch(batch: dict, renderer, context, transparent: bool) -> None:
    """Create Metal buffers from a batch dict and call renderer.renderTriangles."""
    try:
        verts = batch["vertices"]
        normals = batch["normals"]
        colors = batch["colors"]
        indices = batch["indices"]

        # Flatten to 1-D byte arrays for MTLBuffer creation.
        v_bytes = verts.tobytes()
        n_bytes = normals.tobytes()
        c_bytes = colors.tobytes()
        i_bytes = indices.tobytes()

        renderer.renderTriangles(v_bytes, n_bytes, c_bytes, i_bytes,
                                 len(indices) * 3, transparent)
    except Exception:
        pass  # Log but don't crash; individual batch failures are non-fatal.


# --------------------------------------------------------------------------
# Geometry helpers — all fp32
# --------------------------------------------------------------------------

def _as_fp32(arr: np.ndarray) -> np.ndarray:
    from .fp32_utils import to_fp32_vertices
    return to_fp32_vertices(arr)


def _compute_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Flat per-vertex normals computed from triangle face normals."""
    normals = np.zeros_like(verts, dtype=np.float32)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    # Accumulate face normals at each vertex.
    for i in range(3):
        np.add.at(normals, tris[:, i], face_normals)
    # Normalize.
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return normals / norms


def _resolve_colors(drawing, verts: np.ndarray) -> np.ndarray:
    """Return a (N,4) float32 RGBA array in 0-1 range."""
    from .fp32_utils import to_fp32_colors
    n = len(verts)

    if drawing.vertex_colors is not None:
        c = to_fp32_colors(drawing.vertex_colors)
        if c.shape[0] == n:
            return c

    rgba = to_fp32_colors(np.asarray(drawing.color))
    return np.broadcast_to(rgba, (n, 4)).copy()


def _get_positions(drawing) -> list:
    """Return a list of 4x4 float32 position matrices for all displayed copies."""
    positions = drawing.get_positions(displayed_only=True)
    if positions is None or len(positions) == 0:
        return [np.eye(4, dtype=np.float32)]
    return [p.matrix.astype(np.float32) for p in positions]


def _transform_points(verts: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine matrix (row-major) to an (N,3) point array."""
    r = mat4[:3, :3]
    t = mat4[:3, 3]
    return (verts @ r.T + t).astype(np.float32)


def _transform_normals(normals: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    """Apply the inverse-transpose of mat4[:3,:3] to normals."""
    r = mat4[:3, :3]
    try:
        inv_r = np.linalg.inv(r).T
    except np.linalg.LinAlgError:
        inv_r = r
    result = (normals @ inv_r.T).astype(np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return result / norms


def _is_transparent(drawing) -> bool:
    """Return True if the drawing has any transparency."""
    c = drawing.color
    if hasattr(c, '__len__') and len(c) == 4:
        if c[3] < 255:
            return True
    if drawing.vertex_colors is not None:
        if drawing.vertex_colors[:, 3].min() < 255:
            return True
    return False
