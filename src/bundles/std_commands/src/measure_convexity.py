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

def measure_convexity(session, surfaces, palette = None, range = None, key = False,
                      smoothing_iterations = 5, write_surface_data = None, patches = None):
    '''
    Compute the convexity at each surface vertex defined as 2*pi minus the cone-angle
    spanned by the triangles incident at the vertex.  The surface vertices are colored
    based on convexity value.

    This definition of convexity is not standard and gives mottled coloring because the
    values depend strongly on the triangulation.  Vertices surrounded by large triangles
    on a smooth surface will have sharper cone angles then vertices surrounded by small
    triangles.  (Normalizing by vertex triangle areas does not help because the patch about
    a vertex is often irregular in shape.) To ameliorate this the "smoothing_iterations"
    option is useful.

    Parameters
    ----------
    surface : Surface list
    palette : Colormap
      Default color palette is cyan-gray-maroon.
    range : 2-tuple of float or "full"
      Minimum and maximum convexity values corresponding to ends of color palette.
    key : bool
      Whether to show a color key.  Default false.
    smoothing_iterations : int
      Convexity values are averaged with neighbor vertices connected by an edge.
      This value specifies how many rounds of smoothing to perform.  Default 5.
    write_surface_data : string
      File path to write a text file containing surface data.  One line for each
      surface vertex gives vertex number (starting at 0), xyz position, normal
      vector, and (smoothed) convexity value. Vertex lines are followed by surface
      triangle data, one line per triangle giving triangle number and 3 vertex
      numbers for the triangle corners.  The vertex data and triangle data begin
      with comment lines starting with the "#" character.  If multiple surfaces
      are specified the vertex/triangle data sections for each surface are appended
      to the file.
    patches : float
      Instead of coloring by convexity value, color each connected patch with convexity
      above the specified value a unique color.
    '''
    if palette is None:
        from chimerax.core import colors
        palette = colors.BuiltinColormaps['cyan-gray-maroon']
    if range is None and not palette.values_specified:
        range = (-1,1)
    if range is not None and range != 'full':
        rmin, rmax = range
        palette = palette.rescale_range(rmin, rmax)
    sd_file = None if write_surface_data is None else open(write_surface_data, 'w')
    for s in surfaces:
        if s.empty_drawing():
            continue
        va,ta = s.vertices, s.triangles
        from chimerax.surface import vertex_convexity
        c = vertex_convexity(va, ta, smoothing_iterations)
        s.convexity = c
        if patches is None:
            cmap = palette.rescale_range(c.min(), c.max()) if range == 'full' else palette
            vc = cmap.interpolated_rgba8(c)
        else:
            vc = s.get_vertex_colors(create = True)
            pv = _patch_vertices(patches, c, ta)
            from chimerax.core.colors import random_colors
            rc = random_colors(len(pv))
            for i,v in enumerate(pv):
                vc[v] = rc[i]
        s.vertex_colors = vc
        if key and patches is None:
            from chimerax.color_key import show_key
            show_key(session, cmap)

        if sd_file:
            sd_file.write(_surface_data(s, c))
        msg = ('Convexity %.3g - %.3g, mean %.3g, std deviation %.3g at %d vertices of %s'
               % (c.min(), c.max(), c.mean(), c.std(), len(va), s.name))
        from chimerax.core.models import Model
        if isinstance(s, Model):
            msg += ' ' + s.id_string
        session.logger.status(msg, log = True)
    if sd_file:
        sd_file.close()
            
def _surface_data(surf, vertex_values):
    surf_name = surf.name
    from chimerax.core.models import Model
    if not isinstance(surf, Model) and surf.parent:
        surf_name = surf.parent.name + ' ' + surf.name
    vertices, normals, triangles = surf.vertices, surf.normals, surf.triangles
    lines = ['# Surface vertex data for %s, %d vertices, %d triangles'
             % (surf_name, len(vertices), len(triangles))]
    lines.append('# vnum x y z nx ny nz c')
    for i, (v, n, c) in enumerate(zip(vertices, normals, vertex_values)):
        x,y,z = v
        nx,ny,nz = n
        lines.append('%d %.6g %.6g %.6g %.6g %.6g %.6g %.6g' % (i, x, y, z, nx, ny, nz, c))
    lines.append('# Surface triangle data for %s, %d vertices, %d triangles'
                 % (surf_name, len(vertices), len(triangles)))
    lines.append('# tnum v1 v2 v3')
    for i, t in enumerate(triangles):
        v1,v2,v3 = t
        lines.append('%d %d %d %d' % (i, v1, v2, v3))
    sdata = '\n'.join(lines)
    return sdata

def _patch_vertices(threshold, vertex_values, triangles):
    '''
    Find connected surface patches with vertex values above a specified threshold.
    Return list of vertex index arrays, one array for each connected patch.
    '''
    vset = set((vertex_values >= threshold).nonzero()[0])
    tabove = [v1 in vset and v2 in vset and v3 in vset for v1,v2,v3 in triangles]
    ta = triangles[tabove]
    from chimerax.surface import connected_pieces
    vpatch = [v for v,t in connected_pieces(ta)]
    return vpatch

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfacesArg, ColormapArg, \
        ColormapRangeArg, IntArg, SaveFileNameArg, FloatArg, BoolArg
    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        keyword = [('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('key', BoolArg),
                   ('smoothing_iterations', IntArg),
                   ('write_surface_data', SaveFileNameArg),
                   ('patches', FloatArg)],
        synopsis = 'compute surface convexity')
    register('measure convexity', desc, measure_convexity, logger=logger)
