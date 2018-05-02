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
def color_radial(session, surfaces, center = None, coordinate_system = None, palette = None, range = None,
                 update = True):
    _color_geometry(session, surfaces, geometry = 'radial', center = center, coordinate_system = coordinate_system,
                    palette = palette, range = range, auto_update = update)

# -----------------------------------------------------------------------------
#
def color_cylindrical(session, surfaces, center = None, axis = None, coordinate_system = None,
                      palette = None, range = None, update = True):
    _color_geometry(session, surfaces, geometry = 'cylindrical',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range, auto_update = update)

# -----------------------------------------------------------------------------
#
def color_height(session, surfaces, center = None, axis = None, coordinate_system = None,
                 palette = None, range = None, update = True):
    _color_geometry(session, surfaces, geometry = 'height',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range, auto_update = update)

# -----------------------------------------------------------------------------
#
def _color_geometry(session, surfaces, geometry = 'radial',
                    center = None, axis = None, coordinate_system = None,
                    palette = 'redblue', range = None,
                    auto_update = True, caps_only = False):
    surfs = [s for s in surfaces if s.vertices is not None]

    c0 = None
    if center:
        c0 = center.scene_coordinates(coordinate_system)
    elif axis:
        c0 = axis.base_point()

    cclass = {'radial': RadialColor,
              'cylindrical': CylinderColor,
              'height': HeightColor}[geometry]
    for surf in surfs:
        # Set origin and axis for coloring
        c = surf.scene_position.origin() if c0 is None else c0
        if axis:
            a = axis.scene_coordinates(coordinate_system, session.main_view.camera)	# Scene coords
        else:
            a = surf.scene_position.z_axis()
        cs = cclass(surf, palette, range, origin = c, axis = a, auto_recolor = auto_update)
        cs.set_vertex_colors()
    
# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class GeometryColor(State):

    menu_name = 'distance'
    uses_volume_data = False
    uses_origin = True
    uses_axis = True

    def __init__(self, surface, palette, range, origin = (0,0,0), axis = (0,0,1), auto_recolor = True):

        self.surface = surface
        self.colormap = None
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
    def set_origin(self, origin):
        if not self.uses_origin:
            return
        if origin is None:
            s = self.surface
            b = s.bounds()
            lc = (0,0,0) if b is None else b.center()
            origin = s.scene_position * lc
        self.origin = tuple(origin)

    # -------------------------------------------------------------------------
    #
    def set_axis(self, axis):
        self.axis = tuple(axis)
        
    # -------------------------------------------------------------------------
    #
    def set_colormap(self, palette, range):
        from .colorvol import _use_full_range, _colormap_with_range
        r = self.value_range() if _use_full_range(range, palette) else range
        self.colormap = _colormap_with_range(palette, r)

    # -------------------------------------------------------------------------
    #
    def color_surface_pieces(self, plist):

        for p in plist:
            p.vertexColors = self.vertex_colors(p)
            p.using_surface_coloring = True

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
        c = cls(surf, palette = data['colormap'], range = None,
                origin = data['origin'], axis = data['axis'])
        c.set_vertex_colors()
        return c
    
# -----------------------------------------------------------------------------
#
class HeightColor(GeometryColor):

    menu_name = 'height'
        
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
        
    from chimerax.core import geometry
    geometry.distances_parallel_to_axis(points, origin, axis, d)

    return d
    
# -----------------------------------------------------------------------------
#
class RadialColor(GeometryColor):

    menu_name = 'radius'
    uses_axis = False
        
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
        
    from chimerax.core.geometry import distances_from_origin
    distances_from_origin(points, origin, d)

    return d

# -----------------------------------------------------------------------------
#
class CylinderColor(GeometryColor):

    menu_name = 'cylinder radius'

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
        
    from chimerax.core import geometry
    geometry.distances_perpendicular_to_axis(points, origin, axis, d)

    return d
