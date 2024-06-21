# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

# Show connected pieces of one surface that are close to another surface.
# The code only uses the vertex positions, so the vertices should be spaced finer
# than the desired distance.

def surface_hide_far_blobs(session, surface, near_surface, distance):
    # Find vertices in surface that are close to other surface
    other_vertices = masked_vertices(near_surface)
    from chimerax.geometry import find_close_points
    i1,i2 = find_close_points(surface.vertices, other_vertices, distance)
    nv = len(surface.vertices)
    from numpy import zeros
    vclose = zeros((nv,), bool)
    vclose[i1] = True

    # Find all vertices in connected blobs where at least one vertex of blob
    # is close to other surface.
    vkeep = zeros((nv,), bool)
    from chimerax.surface import connected_pieces
    for vi, ti in connected_pieces(surface.masked_triangles):
        if vclose[vi].any():
            vkeep[vi] = True

    # Find triangles of surface where all vertices are in a close blob.
    tmask = vkeep[surface.triangles[:,0]]

    # Set triangle mask for surface to show only close blobs to other surface.
    surface.triangle_mask = tmask

def masked_vertices(surface):
    if surface.triangle_mask is None:
        return surface.vertices
    mt = surface.masked_triangles
    from numpy import zeros
    vmask = zeros(len(surface.vertices), bool)
    for a in (0,1,2):
        vmask[mt[:,a]] = True
    return surface.vertices[vmask,:]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, FloatArg
    desc = CmdDesc(
        required = [('surface', SurfaceArg)],
        keyword = [('near_surface', SurfaceArg),
                   ('distance', FloatArg)],
        required_arguments = ['near_surface', 'distance'],
        synopsis = 'Hide surface blobs far from another surface'
    )
    register('surface hidefarblobs', desc, surface_hide_far_blobs, logger=logger)
