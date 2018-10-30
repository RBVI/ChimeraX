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
        m = v.matrix(step = step, subregion = subregion)
        from numpy import float64, sqrt
        mean = m.mean(dtype=float64)
        sd = m.std(dtype=float64)
        rms = sqrt(sd*sd + mean*mean)
        msg = ('Map %s, minimum %.4g, maximum %.4g, mean %.4g, SD %.4g, RMS %.4g'
               % (v.name_with_id(), m.min(), m.max(), mean, sd, rms))
        session.logger.status(msg, log=True)


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
