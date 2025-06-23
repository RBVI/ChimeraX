# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def rmsd(session, mgr, **kw):
    s = kw['structure']
    atoms = kw['atoms']
    ref = kw["ref_frame"]
    cur_cs_id = s.active_coordset_id
    s.active_coordset_change_notify = False
    s.active_coordset_id = ref
    ref_coords = atoms.coords
    values = {}
    from chimerax.geometry import align_points
    try:
        for i, cs_id in enumerate(s.coordset_ids):
            s.active_coordset_id = cs_id
            xform, rmsd = align_points(atoms.coords, ref_coords)
            values[cs_id] = rmsd
    finally:
        s.active_coordset_id = cur_cs_id
        s.active_coordset_change_notify = True
    return values

def sasa(session, mgr, **kw):
    s = kw['structure']
    atoms = kw['atoms']
    categories = set(atoms.structure_categories)
    from chimerax.atomic import Atoms
    full_atom_set = Atoms([a for a in s.atoms if a.structure_category in categories])
    from math import log2
    status_frequency = max(1, 250000 // len(full_atom_set))
    cur_cs_id = s.active_coordset_id
    s.active_coordset_change_notify = False
    values = {}
    # Emulate the behavior of chimerax.surface.measure_sasacmd.measure_sasa, but without the logging
    r = full_atom_set.radii
    r += 1.4
    from chimerax.surface import spheres_surface_area
    try:
        for i, cs_id in enumerate(s.coordset_ids):
            s.active_coordset_id = cs_id
            areas = spheres_surface_area(full_atom_set.coords, r)
            a = areas[full_atom_set.mask(atoms)]
            sarea = a.sum()
            values[cs_id] = sarea
            if (i+1) % status_frequency == 0:
                session.logger.status("Computed SASA for %d of %d frames" % (i+1, s.num_coordsets))
    finally:
        s.active_coordset_id = cur_cs_id
        s.active_coordset_change_notify = True
        session.logger.status("")
    return values
