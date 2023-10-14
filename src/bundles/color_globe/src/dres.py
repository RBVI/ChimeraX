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

def read_directional_resolution(session, stream, *args, colormap=None):
    """Show directional resolution on a colored sphere."""

    lines = stream.readlines()
    stream.close()
    xyzres = [[float(v) for v in line.split()] for line in lines if not line.startswith('#')]
    xyz = [(x,y,z) for x,y,z,r in xyzres]
    res = [r for x,y,z,r in xyzres]

    # Figure out colormap from argument, file comment, or default colors
    if colormap is None:
        if lines:
            colormap = colormap_from_comment_line(lines[0])
        if colormap is None:
            from chimerax.core.colors import BuiltinColormaps
            colormap = BuiltinColormaps['rainbow']
    if not colormap.values_specified:
        colormap = colormap.rescale_range(min(res), max(res))


    cg = ColorGlobe(session, xyz, res, colormap,
                    sphere_radius = 0.2,
                    offset = (-.6,-.7,-.7))
    session.main_view.add_overlay(cg)

    ck = ColorKey(colormap, length = 0.3, thickness = 0.05, vertical = True,
                  offset = (-0.9, -0.85, -0.99))
    session.main_view.add_overlay(ck)

    rmin, rmax = colormap.value_range()
    from chimerax.label import label_create
    label_create(session, 'resmin', text = '%.1f' % rmin, xpos = .05, ypos = .05, size = 20)
    label_create(session, 'resmax', text = '%.1f' % rmax, xpos = .05, ypos = .23, size = 20)

    # TODO: Make this a real model so shown in model panel, can be closed, hidden, ...
    #       Also 2d labels should be real models.  Need support for real models to
    #       use positioning in overlay space instead of scene space, not be included
    #       in bounding box, not use shadows.
    # TODO: Currently no way to close globe once it is displayed.

    return [], 'Read %d resolution values' % len(xyzres)

def colormap_from_comment_line(line):
    if not line.startswith('# Colormap '):
        return None
    cf = line[11:].split(',')
    if len(cf) < 2:
        return None
    res = []
    colors = []
    from chimerax.core.colors import BuiltinColors, Colormap
    for c in cf:
        ca = c.split('=')
        if len(ca) != 2:
            return None
        try:
            res.append(float(ca[0].strip()))
        except ValueError:
            return None
        colorname = ca[1].strip()
        if colorname not in BuiltinColors:
            return None
        colors.append(BuiltinColors[colorname])
    cmap = Colormap(res, colors)
    return cmap

from chimerax.graphics.drawing import Drawing
class ColorGlobe(Drawing):
    '''
    Create discs tangent to the surface of a sphere at specified points
    colored using a colormap and value associated with each point.
    This color globe rotates with the camera but does not scale or translate.
    It is intended to be placed in the corner and show directional information
    such as anisotropic density map resolution.
    '''
    def __init__(self, session, sphere_points, values, colormap,
                 sphere_radius = 0.5,       # In screen [-1,1] units
                 offset = (0,0,0),          # In [-1,1] screen space
                 num_disc_points = 8,       # points on circumference of disc
                 disc_radius = None,
                 ):

        self.session = session
        Drawing.__init__(self, 'Color globe')

        self.offset = offset

        if disc_radius is None:
            np = len(sphere_points)
            disc_radius = self.default_disc_radius(np, sphere_radius)

        va, na, ta = self.sphere_geometry(sphere_points, sphere_radius,
                                          disc_radius, num_disc_points)
        vc = self.sphere_colors(values, colormap, num_disc_points)

        self.set_geometry(va, na, ta)
        self.vertex_colors = vc

    def default_disc_radius(self, np, sphere_radius):
        from math import sqrt
        return sphere_radius*0.0055*sqrt(np)

    def sphere_geometry(self, points, sphere_radius, disc_radius, num_disc_points):
        'Place a disc tangent to a unit sphere centered at each point'

        # Disc geometry
        from math import sin, cos, pi
        from numpy import array, float32, int32, concatenate
        r = disc_radius
        n = num_disc_points
        cp = [(r*cos(a*2*pi/n), r*sin(a*2*pi/n), sphere_radius) for a in range(n)]
        dv = array([(0,0,sphere_radius)] + cp, float32)
        dn = array([(0,0,1)]*(n+1), float32)
        dt = array([(0,1+i,1+(i+1)%n) for i in range(n)], int32)

        # Discs for each sphere point
        vs = []
        ns = []
        ts = []
        from chimerax.geometry import vector_rotation, normalize_vector
        for i in range(len(points)):
            p = points[i]
            rv = vector_rotation((0,0,1), p)
            vs.append(rv * dv)
            ns.append(rv * dn)
            ts.append(dt + i*(n+1))
        va, na, ta = concatenate(vs), concatenate(ns), concatenate(ts)
        return va, na, ta

    def sphere_colors(self, values, colormap, num_disc_points):
        'Place a disc tangent to a sphere centered at each point'
        # Colors for disc vertices
        rgba = colormap.interpolated_rgba8(values)
        np = len(values)
        nd = num_disc_points+1
        from numpy import empty, uint8
        ca = empty((nd*np,4), uint8)
        for i in range(nd):
            ca[i::nd,:] = rgba
        return ca

    def draw(self, renderer, draw_pass):
        # TODO: Overlay drawing sets projection to identity which will defeat supersample
        #       image capture using pixel shifts that go into the projection matrix from the camera.
        r = renderer
        # Enable backface culling if discs have spaces between them so far side of
        # globe is not visible.
        r.enable_backface_culling(True)
        # TODO: Enabling depth test causes globe to vanish if silhouette edges turned on.  Why?
        # TODO: Maybe overlay drawing should always use a cleared depth buffer, so
        #       overlay cannot be obscured by real models?
        # Enable depth test so overlapped discs don't look like fish scales dependent
        # on order of drawing of discs.
        r.enable_depth_test(True)
        ox, oy, oz = self.offset
        w, h = r.render_size()
        whmax = max(w,h)
        sx, sy = h/whmax, w/whmax
        r.set_projection_matrix(((sx, 0, 0, 0), (0, sy, 0, 0), (0, 0, 1, 0),
                                 (ox, oy, oz, 1)))
        rot = self.session.main_view.camera.position.zero_translation()
        r.set_view_matrix(rot.inverse())
        Drawing.draw(self, renderer, draw_pass)
        # Restore drawing settings so other overlays get expected state.
        from chimerax.geometry import identity
        r.set_view_matrix(identity())
        r.set_projection_matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                                 (0, 0, 0, 1)))
        r.enable_depth_test(False)
        r.enable_backface_culling(False)

class ColorKey(Drawing):
    '''
    Display a rectangular color key as an overlay in the graphics window.
    It does not include numeric labels.
    '''
    def __init__(self, colormap = None,
                 vertical = True,
                 length = 1, thickness = 0.1, # In screen [-1,1] units
                 offset = (-0.05,-.7,-.7),    # In [-1,1] screen space
                 divisions = 50,
                 ):

        Drawing.__init__(self, 'Color key')

        if colormap is None:
            from chimerax.core.colors import BuiltinColormaps
            colormap = BuiltinColormaps['rainbow']

        va, na, vc, ta = self.rectangle_geometry(length, thickness, vertical, offset, colormap, divisions)
        self.set_geometry(va, na, ta)
        self.vertex_colors = vc

    def rectangle_geometry(self, length, thickness, vertical, offset, colormap, div = 50):
        n = div
        from numpy import empty, zeros, float32, int32, linspace, concatenate
        va = empty((2*n,3), float32)
        ox,oy,oz = offset
        al,at = (1,0) if vertical else (0,1)
        va[:n,al] = va[n:2*n,al] = linspace(0,length,n)
        va[:n,at] = 0
        va[n:2*n,at] = thickness
        va[:,2] = 0
        va[:,:] += offset
        na = zeros((2*n,3), float32)
        na[:,2] = 1
        ta = empty((2*(n-1),3), int32)
        for i in range(n-1):
            ta[i,:] = (i,n+i+1,n+i)
            ta[n-1+i,:] = (i,i+1,n+i+1)
        if vertical:
            t1 = ta[:,1].copy()
            ta[:,1] = ta[:,2]
            ta[:,2] = t1
        vmin, vmax = colormap.value_range()
        rgba = colormap.interpolated_rgba8(linspace(vmin,vmax,n))
        ca = concatenate((rgba, rgba))
        return va, na, ca, ta
