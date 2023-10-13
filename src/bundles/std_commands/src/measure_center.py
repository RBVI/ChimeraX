# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def measure_center(session, objects, level = None, mark = False, color = None,
                   radius = None, name = None, model_id = None):

    mlist = objects.models
    from chimerax.map import Volume
    vlist = [m for m in mlist if isinstance(m, Volume)]
    atoms = objects.atoms
    from chimerax.core.models import Surface
    from chimerax.map.volume import VolumeSurface
    surfs = [s for s in objects.models
             if isinstance(s, Surface) and not isinstance(s, VolumeSurface)]
    if len(vlist) == 0 and len(atoms) == 0 and len(surfs) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volume or atoms or surfaces specified')

    rgba = (180,180,180,255) if color is None else color.uint8x4()
    log = session.logger
    
    for v in vlist:
        ijk = volume_center_of_mass(v, level)
        xyz = v.data.ijk_to_xyz(ijk)
        msg = ('Center of mass grid index for %s = (%.2f, %.2f, %.2f)'
               % (v.name, ijk[0], ijk[1], ijk[2]))
        log.status(msg, log = True)
        txyz = v.scene_position * xyz
        msg = ('Center of mass xyz scene coordinates for %s = (%.2f, %.2f, %.2f)'
               % (v.name, txyz[0], txyz[1], txyz[2]))
        log.status(msg, log = True)
        if mark:
            r = max(v.data.step) if radius is None else radius
            mname = v.name + ' center' if name is None else name
            place_marker(session, txyz, rgba, r, mname, model_id)

    if len(atoms) > 0:
        xyz = atoms_center_of_mass(atoms)
        msg = ('Center of mass of %d atoms = (%.2f, %.2f, %.2f)'
               % (len(atoms), xyz[0], xyz[1], xyz[2]))
        log.status(msg, log = True)
        if mark:
            r = atoms[0].radius if radius is None else radius
            mname = atoms_center_model_name(atoms) if name is None else name
            place_marker(session, xyz, rgba, r, mname, model_id)

    for surf in surfs:
        from chimerax.surface import vertex_areas
        areas = vertex_areas(surf.vertices, surf.triangles)
        area = areas.sum()
        if area == 0:
            msg = 'Surface %s has zero area' % surf
            log.status(msg, log = True)
        else:
            sxyz = (surf.vertices * areas.reshape((len(areas),1))).sum(axis = 0) / area
            xyz = surf.scene_position * sxyz
            msg = ('Center of area of surface %s = (%.2f, %.2f, %.2f)'
                   % (surf, xyz[0], xyz[1], xyz[2]))
            log.status(msg, log = True)
            if mark:
                from math import sqrt
                r = (0.01*sqrt(area)) if radius is None else radius
                mname = surf.name + ' center' if name is None else name
                place_marker(session, xyz, rgba, r, mname, model_id)

# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
# Returns center map index coordinates.
#
def volume_center_of_mass(v, level = None):

    if level is None:
        # Use lowest displayed contour level.
        level = v.minimum_surface_level

    # Get 3-d array of map values.
    m = v.data.full_matrix()

    # Find indices of map values above displayed threshold.
    kji = m.nonzero() if level is None else (m >= level).nonzero()

    # Compute total mass above threshold.
    values = m[kji]
    msum = values.sum()

    # Compute mass-weighted center
    center = [(i*values).sum()/msum for i in kji]
    center.reverse()        # k,j,i -> i,j,k index order

    return center

# -----------------------------------------------------------------------------
# Compute center of mass of atoms.
#
def atoms_center_of_mass(atoms):

    xyz = atoms.scene_coords
    weights = atoms.elements.masses
    w = weights.sum()
    from numpy import dot
    c = tuple(dot(weights,xyz)/w)
    return c

# -----------------------------------------------------------------------------
#
def atoms_center_model_name(atoms):

    mols = atoms.unique_structures
    if len(mols) == 1:
        m0 = mols[0]
        name = m0.name + ' center'
        if len(atoms) != len(m0.atoms):
            name += ' %d atoms' % (len(atoms),)
    else:
        name = 'center %d atoms' % (len(atoms),)
    return name

# -----------------------------------------------------------------------------
#
def place_marker(session, xyz, color, radius, model_name, model_id = None):

    # Locate specified marker model
    mset = None
    from chimerax.markers import MarkerSet
    if model_id is not None:
        msets = session.models.list(model_id = model_id, type = MarkerSet)
        if len(msets) >= 1:
            mset = msets[0]

    if mset is None:
        # Create a new marker model
        mset = MarkerSet(session, model_name)
        mset.id = model_id
        session.models.add([mset])

    # Place marker at center position.
    mset.create_marker(xyz, color, radius)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, FloatArg, BoolArg, ColorArg, \
        StringArg, ModelIdArg
    desc = CmdDesc(
        required = [('objects', ObjectsArg)],
        keyword = [('level', FloatArg),
                   ('mark', BoolArg),
                   ('color', ColorArg),
                   ('radius', FloatArg),
                   ('name', StringArg),
                   ('model_id', ModelIdArg),],
        synopsis = 'measure center of mass')
    register('measure center', desc, measure_center, logger=logger)
