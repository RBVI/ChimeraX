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
def surface_area(varray, tarray):
    '''
    Return the surface area of a triangulation specified by vertex
    and triangle arrays.
    '''
    from ._surface import surface_area
    area = surface_area(varray, tarray)
    return area

# -----------------------------------------------------------------------------
#
def enclosed_volume(varray, tarray):
    '''
    Return the enclosed volume of a surface triangulation specified by vertex
    and triangle arrays. Also returns the number of holes in the surface,
    defined as the number of boundary curves.
    '''
    from ._surface import enclosed_volume
    vol, hole_count = enclosed_volume(varray, tarray)
    if vol < 0:
        # Surface has boundary but edges are traversed in opposing directions.
        return None, hole_count
    return vol, hole_count

# -----------------------------------------------------------------------------
# Calculate volume enclosed by a surface and surface area.
#
def surface_volume_and_area(model):
    '''
    Return the surface area, enclosed volume and number of holes (i.e. boundary
    curves) of surface triangulations specified by vertex and triangle arrays.
    All triangles are used even if the surface is masked or clipped.
    All child models are included.  Only Surface models are included.
    '''
    volume = holes = area = 0
    from chimerax.core.models import Surface
    for d in model.all_models():
        if isinstance(d, Surface) and not getattr(d, 'is_clip_cap', False):
            varray = d.vertices
            tarray = d.joined_triangles if hasattr(d, 'joined_triangles') else d.triangles
            if varray is not None and tarray is not None:
                v, hc = enclosed_volume(varray, tarray)
                volume += 0 if v is None else v
                holes += hc
                area += surface_area(varray, tarray)
    return volume, area, holes

# -----------------------------------------------------------------------------
#
def measure_volume(session, surfaces, include_masked = True, return_holes = False):
    vtot = 0
    totholes = 0
    lines = []
    for surf in surfaces:
        va = surf.vertices
        # Use joined triangles for molecular surfaces with sharp edges that use disconnected triangles.
        ta = surf.joined_triangles if hasattr(surf, 'joined_triangles') else surf.triangles
        if va is None or ta is None:
            v, nholes = 0, 0
        else:
            if not include_masked:
                tmask = surf.triangle_mask
                if tmask is not None:
                    ta = ta[tmask]
            v, nholes = enclosed_volume(va, ta)
        if v is None:
            lines.append('Surface %s (#%s) has boundary edges traversed in opposing directions.'
                         '  Cannot determine volume.' % (surf.name, surf.id_string))
        else:
            vtot += v
            totholes += nholes
            line = 'Enclosed volume for %s (#%s) = %.4g' % (surf.name, surf.id_string, v)
            if nholes > 0:
                line += ' with %d surface holes' % nholes
            lines.append(line)
    if len(surfaces) > 1:
        line = 'Total enclosed volume for %d surfaces = %.4g' % (len(surfaces), vtot)
        if totholes > 0:
            line += ' with %d surface holes' % totholes
        lines.append(line)
    elif len(surfaces) == 0:
        lines.append('No surfaces specified')
    
    msg = '\n'.join(lines)
    if len(lines) == 1:
        session.logger.status(msg)
    session.logger.info(msg)

    return (vtot, totholes) if return_holes else vtot

# -----------------------------------------------------------------------------
#
def measure_area(session, surfaces, include_masked = True):
    atot = 0
    lines = []
    for surf in surfaces:
        va = surf.vertices
        ta = surf.triangles if include_masked else surf.masked_triangles
        if va is None or ta is None:
            a = 0
        else:
            a = surface_area(va, ta)
        atot += a
        lines.append('Surface area for %s (#%s) = %.4g' % (surf.name, surf.id_string, a))
    if len(surfaces) > 1:
        lines.append('Total surface area for %d surfaces = %.4g' % (len(surfaces), atot))
    elif len(surfaces) == 0:
        lines.append('No surfaces specified')
    msg = '\n'.join(lines)
    if len(lines) == 1:
        session.logger.status(msg)
    session.logger.info(msg)
    return atot
        
# -----------------------------------------------------------------------------
#
def register_measure_subcommand(command_name, logger):
    from chimerax.core.commands import register, CmdDesc, SurfacesArg, BoolArg

    if command_name == 'measure volume':
        desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                       keyword = [('include_masked', BoolArg)],
                       synopsis = "measure volume enclosed by surface")
        register('measure volume', desc, measure_volume, logger=logger)

    elif command_name == 'measure area':
        desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                       keyword = [('include_masked', BoolArg)],
                       synopsis = "measure area of surface")
        register('measure area', desc, measure_area, logger=logger)

    elif command_name == 'measure sasa':
        # Register "measure sasa" command
        from . import measure_sasacmd
        measure_sasacmd.register_command(logger)
