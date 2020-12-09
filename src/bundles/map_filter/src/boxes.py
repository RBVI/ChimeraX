# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# ------------------------------------------------------------------------------
# Make separate volumes of a specified size around markers.
#
def boxes(session, volume, atoms, size = 0, isize = None, use_atom_size = False,
          step = None, subregion = None, base_model_id = None):

    vlist = []
    vxfinv = volume.position.inverse()
    ijk_rmin, ijk_rmax = volume.ijk_bounds(step, subregion, integer = True)
    for i,a in enumerate(atoms):
        center = vxfinv * a.scene_coord
        r = 0.5*size
        if use_atom_size:
            r += a.radius
        cubify = True if isize is None else isize
        ijk_min, ijk_max, ijk_step = volume.bounding_region([center], r, step, cubify = cubify)
        ijk_min = [max(s,t) for s,t in zip(ijk_min, ijk_rmin)]
        ijk_max = [min(s,t) for s,t in zip(ijk_max, ijk_rmax)]
        region = (ijk_min, ijk_max, ijk_step)
        from chimerax.map.volume import is_empty_region
        if is_empty_region(region):
            continue
        from chimerax.map_data import GridSubregion
        g = GridSubregion(volume.data, *region)
        g.name = 'box %s' % str(a)
        if base_model_id is None:
            mid = None
        elif len(atoms) == 1:
            mid = base_model_id
        else:
            mid = base_model_id + (i+1,)
        from chimerax.map import volume_from_grid_data
        v = volume_from_grid_data(g, session, model_id = mid, show_dialog = False,
                                  open_model = False)
        v.copy_settings_from(volume, copy_region = False,
                             copy_active = False, copy_zone = False)
        vlist.append(v)
    if vlist:
        if len(atoms) > 1:
            session.models.add_group(vlist, '%d boxes' % len(vlist), id = base_model_id)
        else:
            session.models.add(vlist)
    return vlist
