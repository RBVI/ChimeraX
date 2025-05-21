# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def surface_hide_far_blobs(session, surface, near_surface, distance, symmetric = True,
                           counts_only = False, nested = False):
    '''
    Show connected pieces of one surface that are close to another surface.
    The code only uses the vertex positions, so the vertices should be spaced finer
    than the desired distance.
    '''
    if symmetric:
        near_count, far_count = \
            surface_hide_far_blobs(session, surface, near_surface, distance,
                                   counts_only = counts_only, nested = nested, symmetric = False)
        near_count2, far_count2 = \
            surface_hide_far_blobs(session, near_surface, surface, distance,
                                   counts_only = counts_only, nested = nested, symmetric = False)
        return near_count, far_count, near_count2, far_count2

    nv = len(surface.vertices)
    from numpy import zeros
    vclose = zeros((nv,), bool)
    
    other_vertices = masked_vertices(near_surface)
    from chimerax.geometry import find_close_points
    i1,i2 = find_close_points(surface.vertices, other_vertices, distance)
    vclose[i1] = True

    # Find all vertices in connected blobs where at least one vertex of blob
    # is close to other surface.
    vkeep = zeros((nv,), bool)
    near_count = 0
    from chimerax.surface import connected_pieces
    blobs = connected_pieces(surface.masked_triangles)
    near_vol = getattr(near_surface, 'volume', None)
    for vi, ti in blobs:
        if vclose[vi].any():
            vkeep[vi] = True
            near_count += 1
        elif nested and near_vol:
            if (near_vol.interpolated_values(surface.vertices[vi]) >= near_surface.level).all():
                vkeep[vi] = True
                near_count += 1

    far_count = len(blobs) - near_count
    session.logger.info(f'Surface #{surface.id_string} has {near_count} near blobs, {far_count} far blobs, {near_count + far_count} total')
    
    if not counts_only:
        # Find triangles of surface where all vertices are in a close blob.
        tmask = vkeep[surface.triangles[:,0]]

        # Set triangle mask for surface to show only close blobs to other surface.
        surface.triangle_mask = tmask

    return near_count, far_count

def masked_vertices(surface):
    if surface.triangle_mask is None:
        return surface.vertices
    vmask = shown_vertex_mask(surface)
    return surface.vertices[vmask,:]

def shown_vertex_mask(surface):
    mt = surface.masked_triangles
    from numpy import zeros
    vmask = zeros(len(surface.vertices), bool)
    for a in (0,1,2):
        vmask[mt[:,a]] = True
    return vmask

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, FloatArg, BoolArg
    desc = CmdDesc(
        required = [('surface', SurfaceArg)],
        keyword = [('near_surface', SurfaceArg),
                   ('distance', FloatArg),
                   ('symmetric', BoolArg),
                   ('counts_only', BoolArg),
                   ('nested', BoolArg)],
        required_arguments = ['near_surface', 'distance'],
        synopsis = 'Hide surface blobs far from another surface'
    )
    register('surface hidefarblobs', desc, surface_hide_far_blobs, logger=logger)
