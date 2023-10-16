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
#
def measure_motion(session, surface, to_map = None, scale = 1, color = None,
                   prickles_model = None, steps = 10):
    '''
    Display surface normal lines showing motion from one map time to the next.

    Parameters
    ----------
    surface : surface Model
    to_map : Volume
      Show prickles on given surfaces that extend to the surface shown for this volume.
    scale : float
      Scale the length of the prickles by this factor.
    color : Color
      Color of the prickles.
    prickles_model : Model
      Add prickles as submodels of this model.  If not specified, they are added as a submodel
      of the specified surface.
    steps : int
      Determine prickle length up to this number times the volume grid spacing (minimum along 3 axes).
      Lengths that are integral multiples of the grid spacing are tested until the first step which
      places the prickle end point outside the currently shown map contour surface.
    '''

    va, na = surface.vertices, surface.normals
    if not va is None and not na is None:
        sp = surface.scene_position
        mp = to_map.scene_position
        vv = va if sp.is_identity() and mp.is_identity() else (mp.inverse() * sp) * va
        step_size = min(to_map.data.step)
        level = to_map.minimum_surface_level
        n = len(va)
        from numpy import ones, zeros, float32, logical_and
        vinside = ones((n,), bool)
        vlen = zeros((n,), float32)
        for step in range(steps):
            s = step*step_size
            mval = to_map.interpolated_values(vv + s*na)
            logical_and(vinside, mval > level, vinside)
            vlen[vinside] = s
        vlen *= scale
        _show_prickles(surface, vlen, color, prickles_model, children = False)
    for d in surface.child_drawings():
        measure_motion(session, d, to_map, scale, color, prickles_model, steps)

def _show_prickles(surface, length = 1, color = None, prickles_model = None, children = True):
    va, na, ta = surface.vertices, surface.normals, surface.triangles
    if not va is None and not na is None:
        n = len(va)
        from numpy import empty, float32, int32, arange
        van = empty((2*n,3), float32)
        van[:n,:] = va
        for a in range(3):
            van[n:,a] = va[:,a] + length*na[:,a]
        tan = empty((n,2), int32)
        tan[:,0] = arange(n)
        tan[:,1] = tan[:,0] + n
        from chimerax.core.models import Model
        pm = Model('prickles', surface.session)
        pm.set_geometry(van, None, tan)
        pm.display_style = pm.Mesh
        pm.color = color.uint8x4() if color else (0,255,0,255)
        pm.use_lighting = False
        if prickles_model:
            p = prickles_model
            pp, sp = prickles_model.scene_position, surface.position
            if not pp.is_identity() or not sp.is_identity():
                (pp.inverse() * sp).transform_points(van, in_place = True)
        else:
            p = surface
        p = prickles_model if prickles_model else surface
        p.add([pm])

    if children:
        for d in drawing.child_drawings():
            show_prickles(d, length, color, prickles_model)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, FloatArg, ColorArg, ModelArg, IntArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('surface', SurfaceArg)],
        keyword = [('to_map', MapArg),
                   ('scale', FloatArg),
                   ('color', ColorArg),
                   ('prickles_model', ModelArg),
                   ('steps', IntArg)],
        required_arguments = ['to_map'],
        synopsis = 'Draw cactus prickles showing surface motion')
    register('measure motion', desc, measure_motion, logger=logger)
