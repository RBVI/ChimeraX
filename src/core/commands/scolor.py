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
# Code for scolor (spatial coloring) command, providing the campabilities of
# the Surface Color dialog.
#
#   Syntax: scolor <surfaceSpec>
#               [volume <vspec>]
#               [gradient <vspec>]
#               [geometry radial|cylindrical|height]
#               [zone <aspec>]
#               [color <color>]
#               [center <x,y,z>|<atomspec>]
#               [axis x|y|z|<x,y,z>|<atomspec,atomspec>]
#               [coordinateSystem <modelid>]
#               [cmap <v,color>:<v,color>:...|rainbow|gray|redblue|cyanmaroon]
#               [cmapRange <min,max>|full]
#		[reverseColors true|false]
#		[colorOutsideVolume <color>]
#               [offset <d>|<d1,d2,n>]
#               [autoUpdate true|false]
#               [capOnly true|false]
#               [perPixel true|false]
#               [range <r>]
#
def scolor(session, atoms = None, color = None, opacity = None, byatom = False,
           per_atom_colors = None, map = None, palette = None, range = None, offset = 0):
    '''
    Color surfaces using a variety of methods, for example, to match nearby
    atom colors, or use a single color, or color by electrostatic potential,
    or color radially.  TODO: Only a few options are currently supported.
    '''
    if byatom:
        ns = color_surfaces_at_atoms(atoms, color = color, opacity = opacity,
                                     per_atom_colors = per_atom_colors)
    elif map is not None:
        ns = color_surfaces_by_map_value(atoms, opacity=opacity, map=map,
                                         palette=palette, range=range, offset=offset)
    elif color is not None or opacity is not None:
        ns = color_surfaces_at_atoms(atoms, color = color, opacity = opacity)
    else:
        ns = 0

    return ns

def register_command(session):
    from . import cli, color, ColorArg, FloatArg, ColormapArg, ColormapRangeArg
    from ..map import MapArg
    _scolor_desc = cli.CmdDesc(
        optional = [('atoms', cli.Or(cli.AtomsArg, cli.EmptyArg)),
                    ('color', ColorArg),],
        keyword = [('opacity', cli.IntArg),
                   ('byatom', cli.NoArg),
                   ('map', MapArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('offset', FloatArg)],
        synopsis = 'color surfaces')
    cli.register('scolor', _scolor_desc, scolor, logger=session.logger)

def color_surfaces_at_atoms(atoms = None, color = None, opacity = None, per_atom_colors = None):
    from .. import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        s.color_atom_patches(atoms, color, opacity, per_atom_colors)
    return len(surfs)

def color_surfaces_by_map_value(atoms = None, opacity = None, map = None,
                                palette = None, range = None, offset = 0):
    from .. import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        v, all_atoms = s.vertices_for_atoms(atoms)
        if v is not None:
            cs = VolumeColor(map, offset = offset)
            _set_color_source_palette(cs, s, palette, range)
            vcolors = s.get_vertex_colors(create = True, copy = True)
            vcolors[v] = cs.vertex_colors(s, s.session.logger.info)[v]
            s.set_vertex_colors_and_opacities(v, vcolors, opacity)
    return len(surfs)

def color_surface_by_map_value(surf, map, palette = None, range = None,
                               offset = None, opacity = None):
    cs = VolumeColor(map, offset = offset)
    _set_color_source_palette(cs, surf, palette, range)
    vcolors = cs.vertex_colors(surf, map.session.logger.info)
    _adjust_opacities(vcolors, opacity, surf)
    surf.vertex_colors = vcolors

def _adjust_opacities(vcolors, opacity, surf):
    if opacity is not None:
        vcolors[:,3] = opacity
    else:
        vcolors[:,3] = surf.color[3] if surf.vertex_colors is None else surf.vertex_colors[:,3]

def surface_vertex_opacities(surf, opacity, vmask, vcolors):
    if opacity is None:
        # Preserve current transparency
        vcolors[vmask,3] = surf.vertex_colors[vmask,3] if surf.vertex_colors is not None else surf.color[3]
    elif opacity != 'computed':
        vcolors[vmask,3] = opacity

def color_electrostatic(session, surfaces, map, palette = None, range = None, offset = 1.4, transparency = None):
    _color_by_map_value(session, surfaces, map, palette = palette, range = range,
                        offset = offset, transparency = transparency)
    
def color_sample(session, surfaces, map, palette = None, range = None, offset = 0, transparency = None):
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
                        offset = offset, transparency = transparency)

def color_gradient(session, surfaces, map, palette = None, range = None, offset = 0, transparency = None):
    _color_by_map_value(session, surfaces, map, palette = palette, range = range,
                        offset = offset, transparency = transparency, gradient = True)

def _color_by_map_value(session, surfaces, map, palette = None, range = None,
                        offset = 0, transparency = None, gradient = False):
    surfs = _surface_drawings(surfaces)
    opacity = None
    if transparency is not None:
        opacity = min(255, max(0, int(2.56 * (100 - transparency))))
    cs_class = GradientColor if gradient else VolumeColor
    for surf in surfs:
        cs = cs_class(map, offset = offset)
        _set_color_source_palette(cs, surf, palette, range)
        vcolors = cs.vertex_colors(surf, map.session.logger.info)
        _adjust_opacities(vcolors, opacity, surf)
        surf.vertex_colors = vcolors

def color_radial(session, surfaces, center = None, coordinate_system = None, palette = None, range = None):
    _color_geometry(session, surfaces, geometry = 'radial', center = center, coordinate_system = coordinate_system,
                    palette = palette, range = range)

def color_cylindrical(session, surfaces, center = None, axis = None, coordinate_system = None,
                      palette = None, range = None):
    _color_geometry(session, surfaces, geometry = 'cylindrical',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range)

def color_height(session, surfaces, center = None, axis = None, coordinate_system = None,
                 palette = None, range = None):
    _color_geometry(session, surfaces, geometry = 'height',
                    center = center, axis = axis, coordinate_system = coordinate_system,
                    palette = palette, range = range)

# -----------------------------------------------------------------------------
#
def _color_geometry(session, surfaces, geometry = 'radial',
                    center = None, axis = None, coordinate_system = None,
                    palette = 'redblue', range = None,
                    auto_update = False, cap_only = False):
    surfs = _surface_drawings(surfaces)

    c0 = None
    if center:
        c0 = center.scene_coordinates(coordinate_system)
    elif axis:
        c0 = axis.base_point()

    for surf in surfs:
        cs = {'radial': RadialColor,
              'cylindrical': CylinderColor,
              'height': HeightColor}[geometry]()
        # Find center and axis for surface
        if cs.uses_origin:
            if c0 is None:
                b = surf.bounds()
                lc = (0,0,0) if b is None else b.center()
                c = surf.scene_position * lc
            else:
                c = c0
            cs.set_origin(c)
        if cs.uses_axis:
            if axis:
                c = session.main_view.camera
                a = axis.scene_coordinates(coordinate_system, c)	# Scene coords
            else:
                a = surf.scene_position.z_axis()
            cs.set_axis(a)

        vrange = lambda: _compute_value_range(surf, cs.value_range, cap_only)
        cm = _colormap_with_range(palette, range, vrange)
        cs.set_colormap(cm)
        
        vc = cs.vertex_colors(surf)
        surf.vertex_colors = vc

# -----------------------------------------------------------------------------
#
def _surface_drawings(surfaces):
    surfs = []
    for s in surfaces:
        if s.vertices is not None:
            surfs.append(s)
        if hasattr(s, 'surface_drawings_for_vertex_coloring'):
            surfs.extend(s.surface_drawings_for_vertex_coloring())
    return surfs

# -----------------------------------------------------------------------------
#
def _set_color_source_palette(color_source, surf, cmap = None, cmap_range = None, color_outside_volume = 'gray',
                              per_pixel = False, cap_only = False):
    vrange = lambda: _compute_value_range(surf, color_source.value_range, cap_only)
    cm = _colormap_with_range(cmap, cmap_range, vrange)
    color_source.set_colormap(cm)
    color_source.per_pixel_coloring = per_pixel

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
#
def _compute_value_range(surf, value_range_func, cap_only):

    v0,v1 = _surface_value_range(surf, value_range_func, cap_only)
    if v0 == None:
        v0,v1 = (0,1)
    return v0,v1

# -----------------------------------------------------------------------------
# Calculate range of surface values.  The value range function takes an
# n by 3 NumPy array of floats and returns a min and max value.
#
def _surface_value_range(surf, value_range_function, caps_only,
                        include_outline_boxes = False):

    vrange = (None, None)
    plist = [surf]
    if caps_only:
        import SurfaceCap
        plist = [p for p in plist if SurfaceCap.is_surface_cap(p)]
    if not include_outline_boxes:
        plist = [p for p in plist if not hasattr(p, 'outline_box')]
    for p in plist:
        grange = value_range_function(p)
        vrange = _combine_min_max(vrange, grange)
    return vrange
    
# -----------------------------------------------------------------------------
#
def _combine_min_max(min_max_1, min_max_2):

    if tuple(min_max_1) == (None, None):
        return min_max_2
    elif tuple(min_max_2) == (None, None):
        return min_max_1
    return (min((min_max_1[0], min_max_2[0])),
            max((min_max_1[1], min_max_2[1])))

# -----------------------------------------------------------------------------
#
class VolumeColor:

    menu_name = 'volume data value'
    volume_name = 'volume'
    uses_volume_data = True
    uses_origin = False
    uses_axis = False

    def __init__(self, volume, offset = 0):

        self.volume = volume
        self.colormap = None
        self.offset = offset
        self.per_pixel_coloring = False
        self.solid = None             # Manages 3D texture

    # -------------------------------------------------------------------------
    #
    def set_volume(self, volume):

        self.volume = volume

#        from chimera import addModelClosedCallback
#        addModelClosedCallback(volume, self.volume_closed_cb)

    # -------------------------------------------------------------------------
    #
    def set_colormap(self, colormap):

        self.colormap = colormap
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
    def vertex_colors(self, surface, report_stats = None):

        values, outside = self.volume_values(surface)
        if report_stats and len(values) > 0:
            report_stats('Map values for surface "%s": minimum %.4g, mean %.4g, maximum %.4g'
                         % (surface.name, values.min(), values.mean(), values.max()))
        cmap = self.colormap
        rgba = interpolate_colormap(values, cmap.data_values, cmap.colors,
                                    cmap.color_above_value_range,
                                    cmap.color_below_value_range)
        if len(outside) > 0:
            set_outside_volume_colors(outside, cmap.color_no_value, rgba)

        from numpy import uint8
        rgba8 = (255*rgba).astype(uint8)
        return rgba8
        
    # -------------------------------------------------------------------------
    #
    def value_range(self, surface_piece):

        if self.volume is None:
            return (None, None)

        values, outside = self.volume_values(surface_piece)
        v = inside_values(values, outside)
        return array_value_range(v)
        
    # -------------------------------------------------------------------------
    #
    def volume_values(self, surface):

        s = surface
        # Transform from surface to volume coordinates
        tf = self.volume.scene_position.inverse() * s.scene_position
        v = s.vertices
        n = s.normals
        return self.offset_values(v, n, tf)

    # -------------------------------------------------------------------------
    #
    def offset_values(self, v, n, xf):

        if self.offset == 0:
            # No offset.
            values, outside = self.vertex_values(v, xf)
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
    def volume_closed_cb(self, volume):

        self.volume = None
        
    # -------------------------------------------------------------------------
    #
    def closed(self):

        return self.volume is None

# -----------------------------------------------------------------------------
#
class ElectrostaticColor(VolumeColor):

    menu_name = 'electrostatic potential'
    volume_name = 'potential'

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
class GeometryColor:

    menu_name = 'distance'
    uses_volume_data = False
    uses_origin = True
    uses_axis = True

    def __init__(self):

        self.colormap = None
        self.origin = (0,0,0)
        self.axis = (0,0,1)

    # -------------------------------------------------------------------------
    #
    def set_origin(self, origin):
        self.origin = tuple(origin)
    def set_axis(self, axis):
        self.axis = tuple(axis)
        
    # -------------------------------------------------------------------------
    #
    def set_colormap(self, colormap):

        self.colormap = colormap

    # -------------------------------------------------------------------------
    #
    def color_surface_pieces(self, plist):

        for p in plist:
            p.vertexColors = self.vertex_colors(p)
            p.using_surface_coloring = True

    # -------------------------------------------------------------------------
    #
    def vertex_colors(self, surface):

        vertices, tarray = surface.geometry
        sp = surface.scene_position
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
    def values(self, vertices):
        raise RuntimeError('Derived class "%s" did not implement values() method' % self.__class__)
        
    # -------------------------------------------------------------------------
    #
    def value_range(self, surface):

        vertices, tarray = surface.geometry
        sp = surface.scene_position
        va = vertices if sp.is_identity() else sp * vertices
        v = self.values(va)
        return array_value_range(v)
        
    # -------------------------------------------------------------------------
    #
    def closed(self):

        return False
    
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
# Stop coloring a surface using a coloring function.
#
def stop_coloring_surface(model):
    
    color_updater.stop_coloring(model)

    import SurfaceCap
    cm = SurfaceCap.cap_model(model, create = False)
    if cm and cm != model:
        color_updater.stop_coloring(cm)
            
# -----------------------------------------------------------------------------
#
class Color_Updater:

    def __init__(self):

        self.models = {}

        import SimpleSession
        import chimera
        chimera.triggers.addHandler(SimpleSession.SAVE_SESSION,
                                    self.save_session_cb, None)
            
    # -------------------------------------------------------------------------
    #
    def auto_recolor(self, model, color_source, caps_only):

        add_callback = not self.models.has_key(model)
        self.models[model] = (color_source, caps_only)
        from Surface import set_coloring_method
        set_coloring_method('surface color', model, self.stop_coloring)
        if add_callback:
            model.addGeometryChangedCallback(self.surface_changed_cb)
            from chimera import addModelClosedCallback
            addModelClosedCallback(model, self.model_closed_cb)
            
    # -------------------------------------------------------------------------
    #
    def stop_coloring(self, model, erase_coloring = True):

        if not model in self.models:
            return

        del self.models[model]
        model.removeGeometryChangedCallback(self.surface_changed_cb)

        # Remove per-vertex coloring.
        plist = [p for p in model.surfacePieces
                 if hasattr(p, 'using_surface_coloring')
                 and p.using_surface_coloring]
        for p in plist:
            if erase_coloring:
                p.vertexColors = None
                p.textureId = 0
                p.textureCoordinates = None
            p.using_surface_coloring = False
            
    # -------------------------------------------------------------------------
    # p is SurfacePiece.
    #
    def surface_changed_cb(self, p, detail):

        if detail == 'removed':
            return

        m = p.model
        (color_source, caps_only) = self.models[m]
        if color_source.closed():
            self.stop_coloring(m)
            return
        if caps_only:
            import SurfaceCap
            if not SurfaceCap.is_surface_cap(p):
                return
        include_outline_boxes = False
        if not include_outline_boxes and hasattr(p, 'outline_box'):
            return
        color_source.color_surface_pieces([p])

    # -------------------------------------------------------------------------
    #
    def model_closed_cb(self, model):

        if model in self.models:
            del self.models[model]
            
    # -------------------------------------------------------------------------
    #
    def surface_coloring(self, surface):

        mtable = self.models
        if surface in mtable:
            color_source, caps_only = mtable[surface]
            if color_source.closed():
                self.stop_coloring(surface, erase_coloring = False)
            else:
                return color_source, caps_only
        return None, None
            
    # -------------------------------------------------------------------------
    #
    def save_session_cb(self, trigger, x, file):

        import session
        session.save_surface_color_state(self.models, file)

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

# -----------------------------------------------------------------------------
#
def colorable_surface_models():

    from chimera import openModels
    from _surface import SurfaceModel
    mlist = openModels.list(modelTypes = [SurfaceModel])
    from SurfaceCap import is_surface_cap
    mlist = [m for m in mlist if not is_surface_cap(m)]

    return mlist
            
# -----------------------------------------------------------------------------
#
def surface_coloring(surface):

    return color_updater.surface_coloring(surface)

# -----------------------------------------------------------------------------
# Unused.
#
def unscolor(surfaces):

    from _surface import SurfaceModel
    surfs = set([s for s in surfaces if isinstance(s, SurfaceModel)])
    if len(surfs) == 0:
        raise CommandError('No surfaces specified')
    import ColorZone
    for s in surfs:
        stop_coloring_surface(s)
        ColorZone.uncolor_zone(s)

# -------------------------------------------------------------------------
# Unused.
#
def texture_surface_piece(p, t, txf, border_color, offset = 0):

    p.textureId = t.texture_id()
    p.useTextureTransparency = ('a' in t.color_mode)
    p.textureModulationColor = t.modulation_rgba()
    p.textureBorderColor = border_color
    va = offset_vertices(p.geometry[0], p.normals, offset)
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
coloring_methods = (RadialColor, CylinderColor, HeightColor,
                    ElectrostaticColor, VolumeColor, GradientColor,
                    )

# -----------------------------------------------------------------------------
#
#color_updater = Color_Updater()
