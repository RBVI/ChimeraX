"""
Trajectory helpers for the animations bundle.

Maps a normalized fraction in [0, 1] onto the active coordset of a
multi-coordset Structure (e.g., a morph trajectory).
"""


def find_morph_trajectory(session, structure=None):
    """Return a Structure with multiple coordsets, or None.

    If `structure` is provided, return it when it has more than one coordset.
    Otherwise scan ``session.models`` for the lowest-id Structure with
    ``num_coordsets > 1``. Returns ``None`` if no candidate is found.
    """
    if structure is not None:
        return structure if getattr(structure, "num_coordsets", 0) > 1 else None

    candidates = [
        m for m in session.models.list()
        if getattr(m, "num_coordsets", 0) > 1
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: m.id)
    return candidates[0]


def interpolate_trajectory_ids(structure, id_a, id_b, fraction):
    """Linearly interpolate between two coordset ids and apply.

    Picks the nearest coordset id to ``idx_a + fraction * (idx_b - idx_a)``
    where indices are positions in ``structure.coordset_ids``. Returns the
    coordset id that was applied, or None if either input id is not present
    on the structure.
    """
    if structure is None or id_a is None or id_b is None:
        return None
    coordset_ids = list(structure.coordset_ids)
    if not coordset_ids:
        return None
    try:
        idx_a = coordset_ids.index(id_a)
        idx_b = coordset_ids.index(id_b)
    except ValueError:
        return None
    f = max(0.0, min(1.0, fraction))
    interp_idx = int(round(idx_a + f * (idx_b - idx_a)))
    interp_idx = max(0, min(len(coordset_ids) - 1, interp_idx))
    cs_id = coordset_ids[interp_idx]
    if structure.active_coordset_id != cs_id:
        structure.active_coordset_id = cs_id
    return cs_id


def apply_trajectory_fraction(structure, fraction):
    """Set ``structure.active_coordset_id`` to the nearest frame for ``fraction``.

    ``fraction`` is clamped to [0, 1] and mapped to the closest coordset id
    via ``round(fraction * (N - 1))``. Returns the coordset id that was
    applied, or None if the structure has no coordsets.
    """
    if structure is None:
        return None
    coordset_ids = sorted(structure.coordset_ids)
    n = len(coordset_ids)
    if n == 0:
        return None
    if n == 1:
        cs_id = coordset_ids[0]
    else:
        f = max(0.0, min(1.0, fraction))
        idx = int(round(f * (n - 1)))
        cs_id = coordset_ids[idx]
    if structure.active_coordset_id != cs_id:
        structure.active_coordset_id = cs_id
    return cs_id
