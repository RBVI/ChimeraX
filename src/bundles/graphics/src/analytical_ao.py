"""
Per-atom ambient occlusion for impostor sphere rendering.

Provides two approaches:

1. **Per-atom AO** (``compute_atom_ao``): Gaussian-weighted neighbor
   density → per-atom scalar baked into vertex colors.  Cheap but
   view-independent (uniform darkening per atom).

2. **Per-fragment sphere-sphere AO** (``build_sphere_ao_buffers``):
   Precomputes neighbor lists on the CPU and packs them into GPU buffers.
   The impostor fragment shader then evaluates exact solid-angle
   occlusion at every pixel.  View-dependent, smooth, artifact-free.
"""

import numpy as np


def compute_atom_ao(coords, radii, cutoff=5.0, strength=0.05):
    """Compute per-atom ambient occlusion factors.

    Uses Gaussian-weighted neighbor density: atoms surrounded by many
    close neighbors get darkened, exposed atoms stay bright.  Avoids
    the solid-angle formula which breaks down when VDW spheres overlap.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Atom center positions.
    radii : ndarray, shape (N,)
        Atom radii.
    cutoff : float
        Maximum distance (Angstroms) to consider neighbors.
    strength : float
        Darkening strength.  Higher = darker buried atoms.

    Returns
    -------
    ao : ndarray, shape (N,), dtype float32
        Per-atom AO factor: 1.0 = fully exposed, ~0.3 = deeply buried.
    """
    from scipy.spatial import cKDTree

    N = len(coords)
    if N == 0:
        return np.ones(0, dtype=np.float32)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    if len(pairs) == 0:
        return np.ones(N, dtype=np.float32)

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    dists = np.linalg.norm(coords[j_idx] - coords[i_idx], axis=1)

    # Gaussian weight: close neighbors contribute more.
    # Sigma is based on the sum of radii so the falloff adapts to
    # atom sizes (large atoms occlude at greater distance).
    sigma_ij = radii[i_idx] + radii[j_idx]
    weights = np.exp(-dists * dists / (2.0 * sigma_ij * sigma_ij))

    # Accumulate weighted density per atom (symmetric).
    density = np.zeros(N, dtype=np.float64)
    np.add.at(density, i_idx, weights)
    np.add.at(density, j_idx, weights)

    # Map density → AO via exponential falloff.
    ao = np.exp(-density * strength).astype(np.float32)
    return ao


def apply_ao_to_colors(colors, ao_factors):
    """Modulate RGBA vertex colors by per-atom AO factors.

    Parameters
    ----------
    colors : ndarray, shape (N, 4), dtype uint8
        Original RGBA colors.
    ao_factors : ndarray, shape (N,), dtype float32
        Per-atom AO (1.0 = no change, 0.0 = black).

    Returns
    -------
    modulated : ndarray, shape (N, 4), dtype uint8
        Colors with RGB darkened by the AO factor.  Alpha is unchanged.
    """
    result = colors.copy()
    scale = ao_factors[:, np.newaxis].astype(np.float32)
    result[:, :3] = np.clip(colors[:, :3] * scale, 0, 255).astype(np.uint8)
    return result


def build_sphere_ao_buffers(coords, radii, cutoff=5.0, max_neighbors=16):
    """Build GPU-ready neighbor buffers for per-fragment sphere-sphere AO.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Atom center positions (model space).
    radii : ndarray, shape (N,)
        Atom radii.
    cutoff : float
        Maximum neighbor distance in Angstroms.
    max_neighbors : int
        Maximum neighbors per atom (closest are kept).

    Returns
    -------
    offsets : ndarray, shape (N, 2), dtype uint32
        Per-atom (start_index, count) into the neighbor array.
    neighbors : ndarray, shape (M, 4), dtype float32
        Packed neighbor data: (x, y, z, radius) for each neighbor entry.
    """
    from scipy.spatial import cKDTree

    N = len(coords)
    if N == 0:
        return (np.zeros((0, 2), dtype=np.uint32),
                np.zeros((0, 4), dtype=np.float32))

    tree = cKDTree(coords)

    # Collect neighbors per atom.
    all_neighbors = []
    offsets = np.zeros((N, 2), dtype=np.uint32)
    current_offset = 0

    for i in range(N):
        idxs = tree.query_ball_point(coords[i], cutoff)
        # Remove self.
        idxs = [j for j in idxs if j != i]
        if len(idxs) == 0:
            offsets[i] = (current_offset, 0)
            continue
        # Sort by distance and keep closest max_neighbors.
        dists = np.linalg.norm(coords[idxs] - coords[i], axis=1)
        order = np.argsort(dists)[:max_neighbors]
        kept = [idxs[k] for k in order]
        count = len(kept)

        offsets[i] = (current_offset, count)
        for j in kept:
            all_neighbors.append((coords[j][0], coords[j][1], coords[j][2],
                                  radii[j]))
        current_offset += count

    if len(all_neighbors) == 0:
        neighbors = np.zeros((1, 4), dtype=np.float32)  # dummy
    else:
        neighbors = np.array(all_neighbors, dtype=np.float32)

    return offsets, neighbors


def enable_sphere_ao(drawing, renderer, cutoff=5.0, max_neighbors=16):
    """Compute sphere-sphere AO neighbor buffers and attach them to a drawing.

    Call this on an AtomsDrawing to enable per-fragment analytical AO.
    The buffers are uploaded to the GPU and the impostor shader uses them
    automatically on the next frame.

    Parameters
    ----------
    drawing : AtomsDrawing
        The impostor sphere drawing (must have positions with shift_and_scale).
    renderer : MetalRender
        The active Metal renderer.
    cutoff : float
        Neighbor search radius in Angstroms.
    max_neighbors : int
        Maximum neighbors per atom.
    """
    sas = drawing.positions.shift_and_scale_array()
    if sas is None:
        return
    coords = sas[:, :3].copy()
    radii = sas[:, 3].copy()

    offsets, neighbors = build_sphere_ao_buffers(coords, radii, cutoff, max_neighbors)

    m = renderer._metal
    ctx = renderer._ctx

    # Upload offset buffer (N × 2 uint32 = 8 bytes per atom).
    offset_data = np.ascontiguousarray(offsets, dtype=np.uint32)
    drawing._ao_offset_buffer = m.create_buffer(context=ctx, data=offset_data)

    # Upload neighbor buffer (M × 4 float32 = 16 bytes per neighbor entry).
    neighbor_data = np.ascontiguousarray(neighbors, dtype=np.float32)
    drawing._ao_neighbor_buffer = m.create_buffer(context=ctx, data=neighbor_data)


def disable_sphere_ao(drawing):
    """Remove sphere-sphere AO buffers from a drawing."""
    drawing._ao_offset_buffer = None
    drawing._ao_neighbor_buffer = None
