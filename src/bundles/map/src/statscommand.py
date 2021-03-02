# vim: set expandtab shiftwidth=4 softtabstop=4:

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
#
def volume_statistics(session, volumes = None, step = None, region = None):
    '''
    Report volume mean, standard deviation, and root mean square value
    for the currently displayed subregion and step size, or the specified
    step and subregion.
    '''
    if volumes is None:
        from . import Volume
        vlist = session.models.list(type = Volume)
    else:
        vlist = volumes

    lines = []
    stats = []
    from .volume import mean_sd_rms
    for v in vlist:
        matrix = v.matrix(step=step, subregion=region)
        mean, sd, rms = mean_sd_rms(matrix)
        stats.append((mean, sd, rms))
        descrip = _subregion_description(v, step, region)
        if descrip:
            descrip = ', ' + descrip
        vstat = ('%s (#%s)%s: mean = %.5g, SD = %.5g, RMS = %.5g' %
                 (v.name, v.id_string, descrip, mean, sd, rms))
        lines.append(vstat)

    msg = '\n'.join(lines)
    session.logger.info(msg)
    if len(lines) == 1:
        session.logger.status(msg)

    return stats

# -----------------------------------------------------------------------------
# Implementation of "volume statistics" command.
#
def register_volume_statistics_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from .mapargs import MapsArg, MapRegionArg, MapStepArg

    vstat_desc = CmdDesc(
        optional = [('volumes', MapsArg)],
        keyword = [('step', MapStepArg),
                   ('region', MapRegionArg)],
        synopsis = 'Report map mean, standard deviation, and root mean square values')
    register('volume statistics', vstat_desc, volume_statistics, logger=logger)

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
# Menu entry acts on selected or displayed maps.
#
def show_map_stats(session):
    from chimerax.shortcuts.shortcuts import run_on_maps
    run_on_maps('volume statistics %s')(session)
