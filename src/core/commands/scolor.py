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
def color_surfaces_at_atoms(atoms = None, color = None, opacity = None, per_atom_colors = None):
    from .. import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        s.color_atom_patches(atoms, color, opacity, per_atom_colors)
    return len(surfs)

# -----------------------------------------------------------------------------
#
def color_surfaces_at_residues(residues, colors, opacity = None):
    atoms, acolors = _residue_atoms_and_colors(residues, colors)
    color_surfaces_at_atoms(atoms, opacity=opacity, per_atom_colors = acolors)

# -----------------------------------------------------------------------------
#
def _residue_atoms_and_colors(residues, colors):
    atoms = residues.atoms
    from numpy import repeat
    acolors = repeat(colors, residues.num_atoms, axis=0)
    return atoms, acolors

# -----------------------------------------------------------------------------
#
def color_surfaces_by_map_value(atoms = None, opacity = None, map = None,
                                palette = None, range = None, offset = 0):
    from .. import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        v, all_atoms = s.vertices_for_atoms(atoms)
        if v is not None:
            cs = VolumeColor(s, map, palette, range, offset = offset)
            vcolors = s.get_vertex_colors(create = True, copy = True)
            vcolors[v] = cs.vertex_colors()[v]
            s.set_vertex_colors_and_opacities(v, vcolors, opacity)
    return len(surfs)

# -----------------------------------------------------------------------------
#
def surface_vertex_opacities(surf, opacity, vmask, vcolors):
    if opacity is None:
        # Preserve current transparency
        vcolors[vmask,3] = surf.vertex_colors[vmask,3] if surf.vertex_colors is not None else surf.color[3]
    elif opacity != 'computed':
        vcolors[vmask,3] = opacity

# -----------------------------------------------------------------------------
#
def color_electrostatic(session, surfaces, map, palette = None, range = None,
                        offset = 1.4, transparency = None, update = True):
    _color_by_map_value(session, surfaces, map, palette = palette, range = range,
                        offset = offset, transparency = transparency, auto_update = update)
    
# -----------------------------------------------------------------------------
#
def color_sample(session, surfaces, map, palette = None, range = None,
                 offset = 0, transparency = None, update = True):
    '''
    Color surfaces using a palette and interpolated map value at each surface vertex.

    surfaces : list of models
      Surfaces to color.
    palette : :class:`.Colormap`
      Color map.
    range : 2 comma-separated floats or "full"
      Specifies the range of map values used for sampling from a palette.
    map : Volume
      Color specified surfaces by sampling from this density map using palette, range, and offset options.
    offset : float
      Displacement distance along surface normals for sampling map when using map option.  Default 0.
    transparency : float
      Percent transparency to use.  If not specified current transparency is preserved.
    '''

    _color_by_map_value(session, surfaces, map, palette = palette, range = range,
                        offset = offset, transparency = transparency, auto_update = update)

# -----------------------------------------------------------------------------
#
def color_gradient(session, surfaces, map, palette = None, range = None,
                   offset = 0, transparency = None, update = True):
    _color_by_map_value(session, surfaces, map, palette = palette, range = range,
                        offset = offset, transparency = transparency, gradient = True, auto_update = update)

# -----------------------------------------------------------------------------
#
def _color_by_map_value(session, surfaces, map, palette = None, range = None,
                        offset = 0, transparency = None, gradient = False, caps_only = False,
                        auto_update = True):
    surfs = _surface_drawings(surfaces, caps_only)
    cs_class = GradientColor if gradient else VolumeColor
    for surf in surfs:
        cs = cs_class(surf, map, palette, range, transparency = transparency,
                      offset = offset, auto_recolor = auto_update)
        cs.set_vertex_colors()

# -----------------------------------------------------------------------------
#
def _adjust_opacities(vcolors, opacity, surf):
    if opacity is not None:
        vcolors[:,3] = opacity
    else:
        vcolors[:,3] = surf.color[3] if surf.vertex_colors is None else surf.vertex_colors[:,3]

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
    surfs = _surface_drawings(surfaces, caps_only)

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
def _surface_drawings(surfaces, caps_only = False, include_outline_boxes = False):
    surfs = []
    for s in surfaces:
        if s.vertices is not None:
            surfs.append(s)
    return surfs

# -----------------------------------------------------------------------------
#
def _colormap_with_range(cmap, cmap_range, value_range, default = 'redblue'):
    if cmap is None:
        from ..colors import BuiltinColormaps
        cmap = BuiltinColormaps[default]
    if cmap_range is None and (cmap is None or not cmap.values_specified):
        cmap_range = 'full'
    if cmap_range is None:
        cm = cmap
    else:
        if cmap_range == 'full':
            vmin,vmax = value_range()
        else:
            vmin,vmax = cmap_range
        if cmap.values_specified:
            cm = cmap.rescale_range(vmin, vmax)
        else:
            cm = cmap.linear_range(vmin, vmax)
    return cm

# -----------------------------------------------------------------------------
# Calculate range of surface values.  The value range function takes an
# n by 3 NumPy array of floats and returns a min and max value.
#
def _compute_value_range(color_source):
    v0,v1 = color_source.value_range()
    if v0 == None:
        v0,v1 = (0,1)
    return v0,v1

# -----------------------------------------------------------------------------
#
from ..state import State
class VolumeColor(State):

    menu_name = 'volume data value'
    volume_name = 'volume'
    uses_volume_data = True
    uses_origin = False
    uses_axis = False

    def __init__(self, surface, volume, palette = None, range = None,
                 transparency = None, offset = 0, auto_recolor = True):

        self.surface = surface
        self.volume = volume
        self.colormap = None
        self.transparency = transparency
        self.offset = offset

        self.per_pixel_coloring = False
        self.solid = None             # Manages 3D texture

        self.set_colormap(palette, range)
        
        if auto_recolor:
            arv = lambda self=self: self.set_vertex_colors(report_stats = False)
        else:
            arv = None
        surface.auto_recolor_vertices = arv

        if auto_recolor:
            _add_color_session_saving(surface.session, self)

    # -------------------------------------------------------------------------
    #
    def set_volume(self, volume):

        self.volume = volume

#        from chimera import addModelClosedCallback
#        addModelClosedCallback(volume, self.volume_closed_cb)

    # -------------------------------------------------------------------------
    #
    def set_colormap(self, palette, range, per_pixel = False):
        vrange = lambda: _compute_value_range(self)
        self.colormap = _colormap_with_range(palette, range, vrange)
        self.per_pixel_coloring = per_pixel
        self.set_texture_colormap()

    # -------------------------------------------------------------------------
    #
    def color_surface_pieces(self, plist):

        t = self.texture()
        if t:
            txf = self.volume.openState.xform
            border_color = self.colormap.color_no_value
            for p in plist:
                texture_surface_piece(p, t, txf, border_color, self.offset)
        else:
            for p in plist:
                p.vertexColors = self.vertex_colors(p)
                p.using_surface_coloring = True
        
    # -------------------------------------------------------------------------
    #
    def vertex_colors(self, report_stats = True):

        values, outside = self.volume_values()
        if values is None:
            return None
        
        if report_stats and len(values) > 0:
            log = self.volume.session.logger.info
            log('Map values for surface "%s": minimum %.4g, mean %.4g, maximum %.4g'
                % (self.surface.name, values.min(), values.mean(), values.max()))
        cmap = self.colormap
        rgba = interpolate_colormap(values, cmap.data_values, cmap.colors,
                                    cmap.color_above_value_range,
                                    cmap.color_below_value_range)
        if len(outside) > 0:
            set_outside_volume_colors(outside, cmap.color_no_value, rgba)

        from numpy import uint8
        rgba8 = (255*rgba).astype(uint8)

        if self.transparency is not None:
            opacity = min(255, max(0, int(2.56 * (100 - self.transparency))))
            rgba8[:,3] = opacity
        else:
            rgba8[:,3] = self.surface.color[3]

        return rgba8
        
    # -------------------------------------------------------------------------
    #
    def set_vertex_colors(self, report_stats = True):
        s = self.surface
        arv = s.auto_recolor_vertices
        s.vertex_colors = self.vertex_colors(report_stats)
        if arv:
            s.auto_recolor_vertices = arv
        
    # -------------------------------------------------------------------------
    #
    def value_range(self):

        if self.volume is None:
            return (None, None)

        values, outside = self.volume_values()
        if values is None:
            return None, None
        v = inside_values(values, outside)
        return array_value_range(v)
        
    # -------------------------------------------------------------------------
    #
    def volume_values(self):

        s = self.surface
        # Transform from surface to volume coordinates
        tf = self.volume.scene_position.inverse() * s.scene_position
        v = s.vertices
        if v is None:
            return None, None
        n = s.normals
        return self.offset_values(v, n, tf)

    # -------------------------------------------------------------------------
    #
    def offset_values(self, v, n, xf):

        if self.offset == 0:
            # No offset.
            values, outside = self.vertex_values(v, xf)
        elif len(n) != len(v):
            # TODO: Normals are out of sync with vertices.
            values, outside = self.vertex_values(v, xf)
            vol = self.volume
            vol.session.logger.info('Warning! Normals for %s are out of sync with vertices so coloring offset is not being used.' % vol.name)
        elif isinstance(self.offset, (tuple, list)):
            # Average values from several offsets.
            val = None
            out = set()
            for o in self.offset:
                vo = offset_vertices(v, n, o)
                values, outside = self.vertex_values(vo, xf)
                if val is None:
                    val = values
                else:
                    val += values
                out.update(set(outside))
            val *= 1.0/len(self.offset)
            values = val
            outside = list(out)
        else:
            # Single offset
            vo = offset_vertices(v, n, self.offset)
            values, outside = self.vertex_values(vo, xf)

        return values, outside

    # -------------------------------------------------------------------------
    #
    def vertex_values(self, vertices, vertex_xform):

        v = self.volume
        values, outside = v.interpolated_values(vertices, vertex_xform,
                                                out_of_bounds_list = True)
        return values, outside

    # -------------------------------------------------------------------------
    #
    def texture(self):

        if not self.per_pixel_coloring:
            return None
        if isinstance(self.offset, (tuple, list)):
            return None
        if self.solid is None:
            self.create_3d_texture()
        t = self.solid.volume
        if t.texture_id() == 0:
            return None
        return t

    # -------------------------------------------------------------------------
    #
    def create_3d_texture(self):

        v = self.volume
        name = v.name + ' texture'
        matrix = v.full_matrix()
        matrix_id = 0
        transform = v.matrix_indices_to_xyz_transform(step = 1,
                                                      subregion = 'all')
        msize = tuple(matrix.shape[::-1])
        def mplane(axis, plane, m=matrix):
            if axis is None:
                return m
        from VolumeViewer import solid
        s = solid.Solid(name, msize, matrix.dtype, matrix_id, mplane, transform)
        self.solid = s
        s.set_options(color_mode = 'auto8',
                      projection_mode = '3d',
                      dim_transparent_voxels = True,
                      bt_correction = False,
                      minimal_texture_memory = False,
                      maximum_intensity_projection = False,
                      linear_interpolation = True,
                      show_outline_box = False,
                      outline_box_rgb = (1,1,1),
                      outline_box_linewidth = 0,
                      box_faces = False,
                      orthoplanes_shown = (False, False, False),
                      orthoplane_mijk = (0,0,0))
        s.use_plane_callback = False
        self.set_texture_colormap()
        return s

    # -------------------------------------------------------------------------
    #
    def set_texture_colormap(self):

        cmap = self.colormap
        s = self.solid
        if cmap and s:
            tfunc = map(lambda v,c: (v,1) + tuple(c),
                        cmap.data_values, cmap.colors)
            s.set_colormap(tfunc, 1, None, clamp = True)
            s.update_drawing(open = False)

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'surface': self.surface,
            'volume': self.volume,
            'colormap': self.colormap,
            'transparency': self.transparency,
            'offset': self.offset,
            'version': 1,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        surf = data['surface']
        if surf is None:
            session.logger.warning('Could not restore coloring on surface because surface does not exist.')
        vol = data['volume']
        if surf is None:
            session.logger.warning('Could not restore coloring on surface %s because volume does not exist.'
                                   % surf.name)
        c = cls(surf, vol, palette = data['colormap'], range = None,
                transparency = data['transparency'], offset = data['offset'])
        c.set_vertex_colors()
        return c
            
    # -------------------------------------------------------------------------
    #
    def volume_closed_cb(self, volume):

        self.volume = None

# -----------------------------------------------------------------------------
#
class GradientColor(VolumeColor):

    menu_name ='volume data gradient norm'

    def vertex_values(self, vertices, vertex_xform):

        v = self.volume
        gradients, outside = v.interpolated_gradients(vertices, vertex_xform,
                                                      out_of_bounds_list = True)
        # Compute gradient norms.
        from numpy import multiply, sum, sqrt
        multiply(gradients, gradients, gradients)
        gnorms2 = sum(gradients, axis=1)
        gnorms = sqrt(gnorms2)
        
        return gnorms, outside

    # -------------------------------------------------------------------------
    #
    def color_surface_pieces(self, plist):

        for p in plist:
            p.vertexColors = self.vertex_colors(p)
            p.using_surface_coloring = True
    
# -----------------------------------------------------------------------------
#
def interpolate_colormap(values, color_data_values, rgba_colors,
                         rgba_above_value_range, rgba_below_value_range):

    from .. import map
    rgba = map.interpolate_colormap(values, color_data_values, rgba_colors,
                                    rgba_above_value_range, rgba_below_value_range)
    return rgba
            
# -----------------------------------------------------------------------------
#
def set_outside_volume_colors(outside, rgba_outside_volume, rgba):

    from .. import map
    map.set_outside_volume_colors(outside, rgba_outside_volume, rgba)
            
# -----------------------------------------------------------------------------
#
def inside_values(values, outside):
        
    if len(outside) == 0:
        return values
    
    n = len(values)
    if len(outside) == n:
        return ()

    from numpy import zeros, int, array, put, subtract, nonzero, take
    m = zeros(n, int)
    one = array(1, int)
    put(m, outside, one)
    subtract(m, one, m)
    inside = nonzero(m)[0]
    v = take(values, inside, axis=0)
    return v
    
# -----------------------------------------------------------------------------
#
def array_value_range(a):

    if len(a) == 0:
        return (None, None)
    from numpy import minimum, maximum
    r = (minimum.reduce(a), maximum.reduce(a))
    return r
    
# -----------------------------------------------------------------------------
#
def offset_vertices(vertices, normals, offset):

    if offset == 0:
        return vertices
    vo = offset * normals
    vo += vertices
    return vo
    
# -----------------------------------------------------------------------------
#
from ..state import State
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
            _add_color_session_saving(surface.session, self)

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
        vrange = lambda: _compute_value_range(self)
        self.colormap = _colormap_with_range(palette, range, vrange)

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
        rgba = interpolate_colormap(values, cmap.data_values, cmap.colors,
                                    cmap.color_above_value_range,
                                    cmap.color_below_value_range)
        from numpy import uint8
        rgba8 = (255*rgba).astype(uint8)
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
        return array_value_range(v)

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
        
    from .. import geometry
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
        
    from ..geometry import distances_from_origin
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
        
    from .. import geometry
    geometry.distances_perpendicular_to_axis(points, origin, axis, d)

    return d

# -----------------------------------------------------------------------------
#
class SurfaceColorers(State):
    def __init__(self):
        from weakref import WeakSet
        self._colorers = WeakSet()

    def add(self, colorer):
        self._colorers.add(colorer)
        
    def take_snapshot(self, session, flags):
        data = {'colorers': tuple(self._colorers),
                'version': 1}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        # Actual colorers are added when each is restored.
        return SurfaceColorers()

    def clear(self):
        self._colorers.clear()
        
# -----------------------------------------------------------------------------
#
def _add_color_session_saving(session, colorer):
    if not hasattr(session, '_surface_vertex_colorings'):
        session._surface_vertex_colorings = SurfaceColorers()
    session._surface_vertex_colorings.add(colorer)

# -----------------------------------------------------------------------------
#
def surface_value_at_window_position(window_x, window_y):

  # Find surfaces being colored by volume data.
  surface_models = [s for s in colorable_surface_models()
                    if surface_coloring(s)[0]]
  if len(surface_models) == 0:
    return None     # No colored surfaces

  # Find surface intercept under mouse
  from VolumeViewer import slice
  xyz_in, xyz_out = slice.clip_plane_points(window_x, window_y)
  import PickBlobs as pb
  f, p, t = pb.closest_surface_intercept(surface_models, xyz_in, xyz_out)
  if f is None:
    return None     # No surface intercept

  # Look-up surface coloring volume value
  from Matrix import linear_combination, apply_matrix, xform_matrix
  xyz = linear_combination((1.0-f), xyz_in, f, xyz_out)
  s = p.model
  sxf = s.openState.xform
  sxyz = apply_matrix(xform_matrix(sxf.inverse()), xyz)
  cs = surface_coloring(s)[0]
  if isinstance(cs, GeometryColor):
      v = cs.values([sxyz])[0]
  elif isinstance(cs, VolumeColor):
      n = interpolated_surface_normal(sxyz, p, t)
      from numpy import array
      v = cs.offset_values(array(sxyz, ndmin=2), array(n, ndmin=2), sxf)[0]
      
  return v, sxyz, cs.menu_name

# -----------------------------------------------------------------------------
# Find interpolated position of point inside triangle and interpolate normal
# at that position.
#
def interpolated_surface_normal(sxyz, p, t):

  va, ta = p.maskedGeometry(p.Solid)
  tri = ta[t]
  tw = triangle_vertex_weights(sxyz, [va[vi] for vi in tri])
  tn = [p.normals[vi] for vi in tri]
  sn = sum([w*n for w,n in zip(tw,tn)])
  return sn

# -----------------------------------------------------------------------------
# Find weights for triangle vertices to add up to given point inside triangle.
#
def triangle_vertex_weights(xyz, triangle_corners):

  c0, c1, c2 = triangle_corners
  p = xyz - c0
  from Matrix import distance, cross_product as cp
  from numpy import dot as ip
  n = cp(c1-c0, c2-c0)
  n1 = cp(n,c1-c0)
  n2 = cp(n,c2-c0)
  i1 = ip(n1,c2-c0)
  a2 = ip(n1,p)/i1 if i1 != 0 else 0
  i2 = ip(n2,c1-c0)
  a1 = ip(n2,p)/i2 if i2 != 0 else 0
  w = ((1-a1-a2),a1,a2)
  return w

# -------------------------------------------------------------------------
# Unused.
#
def texture_surface_piece(p, t, txf, border_color, offset = 0):

    p.textureId = t.texture_id()
    p.useTextureTransparency = ('a' in t.color_mode)
    p.textureModulationColor = t.modulation_rgba()
    p.textureBorderColor = border_color
    va = offset_vertices(p.vertices, p.normals, offset)
    s2tc = t.texture_matrix()
    p2s = p.model.openState.xform
    p2s.premultiply(txf.inverse())
    import Matrix
    p2tc = Matrix.multiply_matrices(s2tc, Matrix.xform_matrix(p2s))
    tc = va.copy() if offset == 0 else va
    import _contour
    _contour.affine_transform_vertices(tc, p2tc)
    p.textureCoordinates = tc
    p.using_surface_coloring = True

# -----------------------------------------------------------------------------
#
coloring_methods = (RadialColor, CylinderColor, HeightColor, VolumeColor, GradientColor)

# -----------------------------------------------------------------------------
#
#color_updater = Color_Updater()
