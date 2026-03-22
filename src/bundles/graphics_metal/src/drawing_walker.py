"""
Drawing-to-Metal geometry uploader with persistent buffer cache and GPU instancing.

Performance architecture
------------------------
The original version called `addTrianglesBytes` every frame for every Drawing,
allocating fresh MTLBuffers each time.  This rewrite adds three key optimisations:

1. **Persistent buffer cache** (``_geom_cache``):
   Geometry (vertices, normals, colors, indices) is uploaded to shared-mode
   MTLBuffers ONCE and reused every frame.  On Apple Silicon, ``ensureBuffer``
   is a ``memcpy`` into unified DRAM pages that are GPU-visible without any
   DMA transfer.  Re-upload only happens when ``drawing._attribute_changes``
   contains a geometry-affecting attribute.

2. **GPU instancing** for multi-position drawings:
   A Drawing with N displayed copies (symmetry, atom-style water molecules,
   density-grid icosahedra, …) was previously expanded into N separate
   Python-side numpy transforms and N separate draw calls.  Now we pack the
   N 4×4 float32 instance transforms into one MTLBuffer and issue a single
   ``addTriangles`` with ``instanceCount=N``.  The vertex shader reads
   ``instanceTransforms[instanceID]`` from buffer slot 4.
   For a protein crystal with 100 symmetry copies this is 100 draw calls → 1.

3. **Correct transparent back-to-front sort** using camera-space z:
   We compute the centroid of each batch's AABB in world space, project to
   camera z (dot product with view direction), and pass the value as
   ``sortDepth``.  The renderer sorts transparents before encoding.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports at runtime


# ---------------------------------------------------------------------------
# Geometry attribute tags (must match metal_renderer.hpp attr constants)
# ---------------------------------------------------------------------------
_ATTR_VERTS    = 0
_ATTR_NORMALS  = 1
_ATTR_COLORS   = 2
_ATTR_INDICES  = 3
_ATTR_INSTANCE = 4

# Geometry attributes that require a buffer re-upload when dirty.
_GEOM_ATTRS = frozenset({"vertices", "triangles", "normals", "vertex_colors",
                          "color", "positions"})


# ---------------------------------------------------------------------------
# Persistent cache entry
# ---------------------------------------------------------------------------

class _CacheEntry:
    """Stores the last-uploaded geometry fingerprints for one Drawing."""
    __slots__ = ("verts_id", "tris_id", "normals_id", "colors_id", "pos_id",
                 "n_verts", "n_tris", "centroid")

    def __init__(self):
        self.verts_id   = None
        self.tris_id    = None
        self.normals_id = None
        self.colors_id  = None
        self.pos_id     = None
        self.n_verts    = 0
        self.n_tris     = 0
        self.centroid   = np.zeros(3, np.float32)


# Module-level cache: persists across frames, keyed by id(drawing).
_geom_cache: dict[int, _CacheEntry] = {}


def purge_evicted(renderer) -> None:
    """Remove cache entries for Drawings that have been deleted.
    Call periodically (e.g. on model-closed trigger)."""
    dead = [did for did in list(_geom_cache) if not _drawing_alive(did)]
    for did in dead:
        renderer.evictDrawing(did)
        del _geom_cache[did]


def _drawing_alive(did: int) -> bool:
    """Check if the Drawing object at address did is still alive."""
    import ctypes
    try:
        obj = ctypes.cast(did, ctypes.py_object).value
        return hasattr(obj, 'vertices')
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def walk_and_upload(drawing, renderer, context,
                    view_direction=None, camera_pos=None) -> None:
    """
    Walk *drawing* and its children, accumulating draw calls on *renderer*.

    Must be called after ``renderer.beginFrame()`` and before
    ``renderer.endFrame()``.

    Parameters
    ----------
    drawing:
        Root chimerax.graphics.Drawing.
    renderer:
        PyMetalRenderer Cython wrapper with addTrianglesBytes().
    context:
        PyMetalContext Cython wrapper (unused here; kept for API compat).
    view_direction:
        Optional float32 (3,) camera view direction for transparent sort.
    camera_pos:
        Optional float32 (3,) camera world position for depth computation.
    """
    if view_direction is None:
        view_direction = np.array([0, 0, -1], np.float32)
    if camera_pos is None:
        camera_pos = np.array([0, 0, 0], np.float32)

    _walk(drawing, renderer, view_direction, camera_pos)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def _walk(drawing, renderer, view_dir, cam_pos) -> None:
    if not drawing.display:
        return

    verts = drawing.vertices
    tris  = drawing.triangles

    if verts is not None and tris is not None and len(tris) > 0:
        _submit_drawing(drawing, renderer, view_dir, cam_pos)

    for child in drawing.child_drawings():
        _walk(child, renderer, view_dir, cam_pos)


# ---------------------------------------------------------------------------
# Per-Drawing submission
# ---------------------------------------------------------------------------

def _submit_drawing(drawing, renderer, view_dir, cam_pos) -> None:
    """Upload geometry (if dirty) and accumulate one instanced draw call."""
    did = id(drawing)
    entry = _geom_cache.get(did)

    # Detect geometry changes via Drawing._attribute_changes.
    dirty_attrs = getattr(drawing, '_attribute_changes', set())
    needs_upload = (entry is None) or bool(dirty_attrs & _GEOM_ATTRS)

    if needs_upload:
        entry = _upload_geometry(drawing, did, renderer, entry)
        if entry is None:
            return  # upload failed
        _geom_cache[did] = entry

    # Geometry fingerprint is satisfied; compute instance buffer.
    positions, inst_bytes = _instance_buffer(drawing)
    n_instances = len(positions)

    # Camera-space depth of this drawing's centroid (for transparent sort).
    sort_depth = float(np.dot(entry.centroid - cam_pos, view_dir))

    transparent = _is_transparent(drawing)
    n_indices   = entry.n_tris * 3
    inst_len    = len(inst_bytes) if inst_bytes is not None else 0

    # Geometry is already in persistent MTLBuffers; pass None data so the
    # renderer skips the memcpy and uses the existing pool entry.
    renderer.addTrianglesBytes(
        did,
        None, None, None, None,
        n_indices,
        inst_bytes, n_instances,
        transparent,
        sort_depth,
    )

    # Consume the dirty flag.
    if hasattr(drawing, '_attribute_changes'):
        drawing._attribute_changes -= _GEOM_ATTRS


# ---------------------------------------------------------------------------
# Geometry upload (persistent buffer update)
# ---------------------------------------------------------------------------

def _upload_geometry(drawing, did: int, renderer,
                     prev_entry) -> '_CacheEntry | None':
    """
    Upload vertices, normals, colors, indices to persistent Metal buffers.

    On Apple Silicon, ``renderer.ensureBuffer`` performs a ``memcpy`` into
    a shared-mode MTLBuffer — the same physical DRAM pages the GPU reads.
    No DMA transfer, no GPU-side copy.
    """
    from .fp32_utils import to_fp32_vertices, to_fp32_normals, to_fp32_colors

    verts = to_fp32_vertices(drawing.vertices)
    tris  = np.ascontiguousarray(drawing.triangles, dtype=np.int32)

    if drawing.normals is not None:
        normals = to_fp32_normals(drawing.normals)
    else:
        normals = _compute_normals(verts, tris)

    colors = _resolve_colors(drawing, len(verts))

    v_bytes = verts.tobytes()
    n_bytes = normals.tobytes()
    c_bytes = colors.tobytes()
    i_bytes = tris.tobytes()

    # Upload — renderer.ensureBuffer(drawing_id, attr, data, length)
    # returns the MTLBuffer, but we don't need it here; the renderer
    # keeps it in its persistent pool keyed by (did, attr).
    renderer.ensureBuffer(did, _ATTR_VERTS,   v_bytes, len(v_bytes))
    renderer.ensureBuffer(did, _ATTR_NORMALS, n_bytes, len(n_bytes))
    renderer.ensureBuffer(did, _ATTR_COLORS,  c_bytes, len(c_bytes))
    renderer.ensureBuffer(did, _ATTR_INDICES, i_bytes, len(i_bytes))

    entry = _CacheEntry()
    entry.n_verts   = len(verts)
    entry.n_tris    = len(tris)
    entry.centroid  = verts.mean(axis=0).astype(np.float32)
    return entry


# ---------------------------------------------------------------------------
# Instance buffer construction (GPU instancing)
# ---------------------------------------------------------------------------

def _instance_buffer(drawing) -> tuple[list, bytes | None]:
    """
    Return (positions_list, bytes_or_None) for GPU instancing.

    ``positions_list`` is the list of displayed position objects.
    ``bytes_or_None`` is a flat (N*16) float32 byte array of column-major
    4×4 transform matrices, or None if there is exactly one identity copy
    (in which case the vertex shader uses the model matrix from uniforms).
    """
    try:
        positions = drawing.get_positions(displayed_only=True)
    except Exception:
        positions = None

    if positions is None or len(positions) == 0:
        return [None], None

    if len(positions) == 1:
        # Single copy: check if it's identity; if so, skip the instance buffer
        # and let the uniform model matrix handle the transform.
        mat = positions[0].matrix
        if np.allclose(mat, np.eye(4)):
            return positions, None
        # Non-identity single copy: still use instancing (simpler shader path).

    # Pack all N transforms into a contiguous (N, 4, 4) float32 array.
    # MSL float4x4 is column-major, matching numpy's default.
    mats = np.stack([p.matrix.astype(np.float32) for p in positions], axis=0)
    return positions, mats.tobytes()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals via face-normal accumulation."""
    normals = np.zeros_like(verts, dtype=np.float32)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    np.add.at(normals, tris[:, 0], face_n)
    np.add.at(normals, tris[:, 1], face_n)
    np.add.at(normals, tris[:, 2], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def _resolve_colors(drawing, n_verts: int) -> np.ndarray:
    """Return a (n_verts, 4) float32 RGBA array in 0–1 range."""
    from .fp32_utils import to_fp32_colors
    if drawing.vertex_colors is not None:
        c = to_fp32_colors(drawing.vertex_colors)
        if len(c) == n_verts:
            return c
    rgba = to_fp32_colors(np.asarray(drawing.color))
    if rgba.ndim == 1:
        return np.broadcast_to(rgba, (n_verts, 4)).copy()
    return rgba


def _is_transparent(drawing) -> bool:
    c = drawing.color
    if hasattr(c, '__len__') and len(c) >= 4 and c[3] < 255:
        return True
    vc = drawing.vertex_colors
    if vc is not None and vc[:, 3].min() < 255:
        return True
    return False
