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

_color_geom_args_doc = '''

    surfaces : list of models
      Surfaces to color.
    center : :class:`.Center`
      Center point for geometric coloring.
    axis : :class:`.Axis`
      Axis vector for cylinder or height coloring.
    coordinate_system : Place
      Transform of center and axis to scene coordinates.
    palette : :class:`.Colormap`
      Color map.
    range : 2 comma-separated floats or "full"
      Specifies the range of map values used for sampling from a palette.
    key : bool
      Whether to show a color key.  Default false.
    update : bool
      Whether to automatically update the surface coloring when the surface shape changes.
'''

# -----------------------------------------------------------------------------
#
def color_radial(session, surfaces, center = None, coordinate_system = None,
                 palette = None, range = None, transparency = None,
                 key = False, update = True):
    '''
    Color surfaces by distance from a center point to each surface vertex
    with distances mapped to colors by a color palette.
    '''

    _color_geometry(session, surfaces, geometry = 'radial',
                    center = center, coordinate_system = coordinate_system,
                    palette = palette, range = range, transparency = transparency,
                    key = key, auto_update = update)

color_radial.__doc__ += _color_geom_args_doc

# -----------------------------------------------------------------------------
#
def color_cylindrical(session, surfaces, center = None, axis = None, coordinate_system = None,
                      palette = None, range = None, transparency = None,
                      key = False, update = True):
    '''
    Color surfaces by distance from a cylinder axis to each surface vertex
    with distances mapped to colors by a color palette.
    '''

    _color_geometry(session, surfaces, geometry = 'cylindrical',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range, transparency = transparency,
                    key = key, auto_update = update)

color_cylindrical.__doc__ += _color_geom_args_doc

# -----------------------------------------------------------------------------
#
def color_height(session, surfaces, center = None, axis = None, coordinate_system = None,
                 palette = None, range = None, transparency = None,
                 key = False, update = True):
    '''
    Color surfaces by distance parallel an axis to each surface vertex
    with distances mapped to colors by a color palette.
    '''

    _color_geometry(session, surfaces, geometry = 'height',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range, transparency = transparency,
                    key = key, auto_update = update)

color_height.__doc__ += _color_geom_args_doc

# -----------------------------------------------------------------------------
#
def _color_geometry(session, surfaces, geometry = 'radial',
                    center = None, axis = None, coordinate_system = None,
                    palette = 'redblue', range = None, transparency = None,
                    key = False, auto_update = True, caps_only = False):
    surfs = [s for s in surfaces if s.vertices is not None]

    c0 = None
    if center:
        c0 = center.scene_coordinates(coordinate_system)
    elif axis:
        c0 = axis.base_point()

    cclass = {'radial': RadialColor,
              'cylindrical': CylinderColor,
              'height': HeightColor}[geometry]
    from chimerax.core.undo import UndoState
    undo_state = UndoState('color %s' % geometry)
    for surf in surfs:
        cprev = surf.color_undo_state
        # Set origin and axis for coloring
        c = surf.scene_position.origin() if c0 is None else c0
        if axis:
            # Scene coords
            a = axis.scene_coordinates(coordinate_system, session.main_view.camera)
        else:
            a = surf.scene_position.z_axis()
        cs = cclass(surf, palette, range, transparency = transparency,
                    origin = c, axis = a, auto_recolor = auto_update)
        cs.set_vertex_colors()
        undo_state.add(surf, 'color_undo_state', cprev, surf.color_undo_state)
        if key:
            from chimerax.color_key import show_key
            show_key(session, cs.colormap)

    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class GeometryColor(State):

    def __init__(self, surface, palette, range, transparency = None,
                 origin = (0,0,0), axis = (0,0,1),
                 auto_recolor = True):

        self.surface = surface
        self.colormap = None
        self.transparency = transparency
        self.origin = origin
        self.axis = axis

        self.set_colormap(palette, range)
        
        arv = self.set_vertex_colors if auto_recolor else None
        surface.auto_recolor_vertices = arv

        if auto_recolor:
            from .updaters import add_updater_for_session_saving
            add_updater_for_session_saving(surface.session, self)

    # -------------------------------------------------------------------------
    #
    def active(self):
        s = self.surface
        return s is not None and s.auto_recolor_vertices == self.set_vertex_colors
    
    # -------------------------------------------------------------------------
    #
    def set_colormap(self, palette, range):
        from .colorvol import _use_full_range, _colormap_with_range
        r = self.value_range() if _use_full_range(range, palette) else range
        self.colormap = _colormap_with_range(palette, r)

    # -------------------------------------------------------------------------
    #
    def vertex_colors(self):

        s = self.surface
        vertices = s.vertices
        if vertices is None:
            return None
        sp = s.scene_position
        va = vertices if sp.is_identity() else sp * vertices
        values = self.values(va)
        cmap = self.colormap
        rgba8 = cmap.interpolated_rgba8(values)
        if self.transparency is not None:
            alpha = min(255, max(0, int(2.56 * (100 - self.transparency))))
            rgba8[:,3] = alpha
        return rgba8
        
    # -------------------------------------------------------------------------
    #
    def set_vertex_colors(self):
        s = self.surface
        arv = s.auto_recolor_vertices
        s.vertex_colors = self.vertex_colors()
        if arv:
            s.auto_recolor_vertices = arv
        
    # -------------------------------------------------------------------------
    #
    def values(self, vertices):
        raise RuntimeError('Derived class "%s" did not implement values() method' % self.__class__)
        
    # -------------------------------------------------------------------------
    #
    def value_range(self):

        s = self.surface
        vertices = s.vertices
        sp = s.scene_position
        va = vertices if sp.is_identity() else sp * vertices
        v = self.values(va)
        from .colorvol import _array_value_range
        r = _array_value_range(v)
        return r

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'surface': self.surface,
            'colormap': self.colormap,
            'transparency': self.transparency,
            'origin': self.origin,
            'axis': self.axis,
            'version': 1,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        surf = data['surface']
        if surf is None:
            session.logger.warning('Could not restore coloring on surface %s because surface does not exist.'
                                   % '.'.join('%d' % i for i in id))
            return None
        c = cls(surf, palette = data['colormap'], range = None, transparency = data.get('transparency'),
                origin = data['origin'], axis = data['axis'])
        c.set_vertex_colors()
        return c

# -----------------------------------------------------------------------------
#
def geometry_coloring(surface):
    '''Return GeometryColor class for surface model if it is being auto colored.'''
    arv = surface.auto_recolor_vertices
    if hasattr(arv, '__self__'):
        gc = arv.__self__  # Instance of a bound method
        if isinstance(gc, GeometryColor):
            return gc
    return None
    
# -----------------------------------------------------------------------------
#
class HeightColor(GeometryColor):

    # -------------------------------------------------------------------------
    #
    def values(self, vertices):

        d = distances_along_axis(vertices, self.origin, self.axis)
        return d

# -----------------------------------------------------------------------------
# Given n by 3 array of points and an axis return an array of distances
# of points along the axis.
#
def distances_along_axis(points, origin, axis):

    from numpy import zeros, single as floatc
    d = zeros(len(points), floatc)
        
    from chimerax import geometry
    geometry.distances_parallel_to_axis(points, origin, axis, d)

    return d
    
# -----------------------------------------------------------------------------
#
class RadialColor(GeometryColor):
        
    # -------------------------------------------------------------------------
    #
    def values(self, vertices):
        d = distances_from_origin(vertices, self.origin)
        return d

# -----------------------------------------------------------------------------
# Given n by 3 array of points and an origin return an array of distances
# of points from origin.
#
def distances_from_origin(points, origin):

    from numpy import zeros, single as floatc
    d = zeros(len(points), floatc)
        
    from chimerax.geometry import distances_from_origin
    distances_from_origin(points, origin, d)

    return d

# -----------------------------------------------------------------------------
#
class CylinderColor(GeometryColor):

    def values(self, vertices):

        d = distances_from_axis(vertices, self.origin, self.axis)
        return d

# -----------------------------------------------------------------------------
# Given n by 3 array of points and an axis return an array of distances
# of points from axis.
#
def distances_from_axis(points, origin, axis):

    from numpy import zeros, single as floatc
    d = zeros(len(points), floatc)
        
    from chimerax import geometry
    geometry.distances_perpendicular_to_axis(points, origin, axis, d)

    return d
