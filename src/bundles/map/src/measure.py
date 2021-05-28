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

# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
#
def volume_center_of_mass(v, level = None, step = None, subregion = None):

    if level is None:
        # Use lowest displayed contour level.
        level = v.minimum_surface_level

    # Get 3-d array of map values.
    m = v.matrix(step = step, subregion = subregion)

    # Find indices of map values above displayed threshold.
    kji = (m >= level).nonzero()

    # Compute total mass above threshold.
    msum = m[kji].sum()

    # Compute mass-weighted center
    mcenter = [(i*m[kji]).sum()/msum for i in kji]
    mcenter.reverse()        # k,j,i -> i,j,k index order

    tf = v.matrix_indices_to_xyz_transform(step, subregion)
    center = tf*mcenter

    return center


# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
#
def measure_map_stats(session, volumes = None, step = None, subregion = None):

    if volumes is None:
        from . import Volume
        volumes = session.models.list(type = Volume)
        
    for v in volumes:
        # Get 3-d array of map values.
        reg = _subregion_description(v, step, subregion)
        if reg:
            reg = ', ' + reg
        m = v.matrix(step = step, subregion = subregion)
        from numpy import float64, sqrt
        mean = m.mean(dtype=float64)
        sd = m.std(dtype=float64)
        rms = sqrt(sd*sd + mean*mean)
        msg = ('Map %s%s, minimum %.4g, maximum %.4g, mean %.4g, SD %.4g, RMS %.4g'
               % (v.name_with_id(), reg, m.min(), m.max(), mean, sd, rms))
        session.logger.status(msg, log=True)

# -----------------------------------------------------------------------------
#
def _subregion_description(v, step = None, region = None):

    dlist = []
    ijk_min, ijk_max, ijk_step = [tuple(ijk) for ijk in v.subregion(step, region)]
    if ijk_step != (1,1,1):
        if ijk_step[1] == ijk_step[0] and ijk_step[2] == ijk_step[0]:
            dlist.append('step %d' % ijk_step[0])
        else:
            dlist.append('step %d %d %d' % ijk_step)
    dmax = tuple([s-1 for s in v.data.size])
    if ijk_min != (0,0,0) or ijk_max != dmax:
        dlist.append('subregion %d,%d,%d,%d,%d,%d' % (ijk_min+ijk_max))
    if dlist:
        return ', '.join(dlist)
    return ''

# -----------------------------------------------------------------------------
#
def register_measure_mapstats_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from .mapargs import MapsArg, MapRegionArg, MapStepArg
    desc = CmdDesc(
        optional = [('volumes', MapsArg)],
        keyword = [('step', MapStepArg),
                   ('subregion', MapRegionArg)],
        synopsis = 'Report map statistics'
    )
    register('measure mapstats', desc, measure_map_stats, logger=logger)

# -----------------------------------------------------------------------------
# Menu entry acts on selected or displayed maps.
#
def show_map_stats(session):
    from chimerax.shortcuts.shortcuts import run_on_maps
    run_on_maps('measure mapstats %s')(session)

# -----------------------------------------------------------------------------
# Interpolate map values at atom positions and assign an atom attribute.
#
def measure_map_values(session, map, atoms, attribute = 'mapvalue'):

    # Get atom positions in volume coordinate system.
    points = atoms.scene_coords
    map.position.inverse().transform_points(points, in_place = True)

    # outside is list of indices for atoms outside map bounds
    values, outside = map.interpolated_values(points, out_of_bounds_list = True)

    # Register atom attribute.
    if len(outside) < len(atoms):
        from chimerax.atomic import Atom
        Atom.register_attr(session, attribute, 'map values', attr_type = float)

    # Set atom attribute values
    outset = set(outside)
    for i, (a,v) in enumerate(zip(atoms, values)):
        if i not in outset:
            setattr(a, attribute, v)

    # Log status message
    if len(outside) == 0:
        msg = ('Interpolated map %s values at %d atom positions,'
               ' min %.4g, max %.4g, mean %.4g, SD %.4g' %
               (map.name_with_id(), len(atoms), values.min(), values.max(),
                values.mean(), values.std()))
    elif len(outside) < len(atoms):
        from numpy import ones, uint8
        inside = ones((len(atoms),), uint8)
        inside[outside] = 0
        v = values[inside]
        msg = ('Interpolated map %s values at %d atom positions (%d outside map bounds),'
               ' min %.4g, max %.4g, mean %.4g, SD %.4g' %
               (map.name_with_id(), len(atoms)-len(outside), len(outside),
                v.min(), v.max(), v.mean(), v.std()))
    else:
        msg = 'All %d atoms oustide map %s bounds' % (len(atoms), map.name_with_id())
    session.logger.status(msg, log=True)

    return values, outside

# -----------------------------------------------------------------------------
#
def register_measure_mapvalues_command(logger):

    from chimerax.core.commands import CmdDesc, register, StringArg
    from .mapargs import MapArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('map', MapArg)],
        keyword = [('atoms', AtomsArg),
                   ('attribute', StringArg)],
        required_arguments = ['atoms'],
        synopsis = 'Report map statistics'
    )
    register('measure mapvalues', desc, measure_map_values, logger=logger)
