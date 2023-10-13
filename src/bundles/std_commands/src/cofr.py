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

def cofr(session, method=None, objects=None, pivot=None, coordinate_system=None,
         show_pivot=None):
    '''
    Set center of rotation method to "front center" or "fixed".  For fixed can
    specify the pivot point as the center of specified displayed objects,
    or as a 3-tuple of numbers and optionally a model whose coordinate system
    is used for 3-tuples.

    Parameters
    ----------
    method : string
      "front center" or "fixed" specifies how the center of rotation point is defined.
    objects : Objects
      Set the method to "fixed" and use the center of the bounding box of these objects
      as the pivot point.
    pivot : 3 floats
      Set the method to "fixed" and used the specified point as center of rotation.
    coordinate_system : Model
      The pivot argument is given in the coordinate system of this model.  If this
      option is not specified then the pivot is in scene coordinates.
    show_pivot : bool or 2 floats
      Whether to draw the center of rotation point in the scene as 3 colored axes.
      If two floats are given, they are axes length and radius of the pivot point indicator
      and the pivot is shown.
    '''
    v = session.main_view
    if not method is None:
        if method == 'frontCenter':
            method = 'front center'
        elif method == 'centerOfView':
            method = 'center of view'
        v.center_of_rotation_method = method

    if not objects is None:
        if objects.empty():
            from chimerax.core.errors import UserError
            raise UserError('No objects specified.')
        disp = objects.displayed()
        b = disp.bounds()
        if b is None:
            from chimerax.core.errors import UserError
            raise UserError('No displayed objects specified.')
        v.center_of_rotation = b.center()

    if not pivot is None:
        p = pivot if coordinate_system is None else coordinate_system.scene_position * pivot
        from numpy import array, float32
        v.center_of_rotation = array(p, float32)

    if show_pivot is not None:
        if isinstance(show_pivot, tuple):
            axis_length, axis_radius = show_pivot
            show_cofr_indicator(session, True, axis_length, axis_radius)
        else:
            show_cofr_indicator(session, show_pivot)

    if method is None and objects is None and pivot is None and show_pivot is None:
        msg = 'Center of rotation: %.5g %.5g %.5g  %s' % (tuple(v.center_of_rotation) + (v.center_of_rotation_method,))
        log = session.logger
        log.status(msg)
        log.info(msg)
        
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, EmptyArg, ObjectsArg, Or
    from chimerax.core.commands import BoolArg, Float2Arg, Float3Arg, ModelArg, create_alias
    methods = ('frontCenter', 'fixed', 'centerOfView')
    desc = CmdDesc(
        optional=[('method', Or(EnumOf(methods), EmptyArg)),
                  ('objects', Or(ObjectsArg, EmptyArg)),
                  ('pivot', Float3Arg)],
        keyword=[('coordinate_system', ModelArg),
                 ('show_pivot', Or(Float2Arg, BoolArg))],
        synopsis='set center of rotation method')
    register('cofr', desc, cofr, logger=logger)
    create_alias('~cofr', 'cofr frontCenter')

def show_cofr_indicator(session, show = True, axis_length = 2.0, axis_radius = 0.05):
    v = session.main_view
    i = [d for d in v.drawing.child_drawings() if isinstance(d, PivotIndicator)]
    if show:
        if len(i) == 0:
            pi = PivotIndicator(session, axis_length, axis_radius)
            v.drawing.add_drawing(pi)
        else:
            for pi in i:
                pi.set_size(axis_length, axis_radius)
    else:
        for pi in i:
            v.drawing.remove_drawing(pi)

from chimerax.graphics import Drawing
class PivotIndicator(Drawing):
    skip_bounds = True
    casts_shadows = False
    pickable = False
    
    def __init__(self, session, axis_length = 2.0, axis_radius = 0.05,
                 axis_colors = [(255,0,0,255),(0,255,0,255),(0,0,255,255)]):
        self._session = session
        self._center = None
        Drawing.__init__(self, 'Pivot indicator')
        self.pickable = False    # Don't set depth in frontCenter mode.
        self._create_geometry(axis_length, axis_radius, axis_colors)
        h = session.triggers.add_handler('graphics update', self._update_position)
        self._update_handler = h

    def delete(self):
        self._session.triggers.remove_handler(self._update_handler)
        super().delete()

    def _update_position(self, *_):
        v = self._session.main_view
        if v.center_of_rotation_method == 'front center':
            # Don't recompute front center rotation point, expensive, distracting.
            center = tuple(v._center_of_rotation)
        else:
            center = tuple(v.center_of_rotation)
        if center != self._center:
            self._center = center
            from chimerax.geometry import Place
            self.position = Place(origin = center)

    def _create_geometry(self, axis_length, axis_radius, axis_colors):
        self.set_size(axis_length, axis_radius)
        self.set_colors(axis_colors)

    def set_size(self, axis_length, axis_radius):
        from chimerax.surface.shapes import cylinder_geometry, cone_geometry
        vaz, naz, taz = cylinder_geometry(radius = axis_radius, height = axis_length)
        vcz, ncz, tcz = cone_geometry(radius = axis_radius * 2, height = axis_length * 0.2, 
                                        caps = True)
        from chimerax.geometry import Place
        vct = Place(origin = (0,0,axis_length/2))
        vcz = vct.transform_points(vcz)
        nv = len(vaz)
        tcz = tcz + nv
        from numpy import concatenate
        vaz = concatenate((vaz, vcz))
        naz = concatenate((naz, ncz))
        taz = concatenate((taz, tcz))
        nv = len(vaz)
        tx = Place(axes = [[0,0,1],[0,-1,0],[1,0,0]])
        vax, nax, tax = tx.transform_points(vaz), tx.transform_vectors(naz), taz.copy() + nv
        ty = Place(axes = [[1,0,0],[0,0,-1],[0,1,0]])
        vay, nay, tay = ty.transform_points(vaz), ty.transform_vectors(naz), taz.copy() + 2*nv

        vc = self.vertex_colors
        self.set_geometry(concatenate((vax,vay,vaz)),
                          concatenate((nax,nay,naz)),
                          concatenate((tax,tay,taz)))
        self.vertex_colors = vc		# Setting geometry clears vertex colors

    def set_colors(self, axis_colors):
        # Axis colors red = x, green = y, blue = z
        from numpy import concatenate, empty, uint8
        nv = len(self.vertices)//3
        cax, cay, caz = empty((nv,4), uint8), empty((nv,4), uint8), empty((nv,4), uint8)
        cax[:], cay[:], caz[:] = axis_colors
        self.vertex_colors = concatenate((cax,cay,caz))
