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
def color_electrostatic(session, surfaces, map, palette = None, range = None,
                        offset = 1.4, transparency = None, update = True):
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
def color_surfaces_by_map_value(atoms = None, opacity = None, map = None,
                                palette = None, range = None, offset = 0):
    from chimerax import atomic
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
def _color_by_map_value(session, surfaces, map, palette = None, range = None,
                        offset = 0, transparency = None, gradient = False, caps_only = False,
                        auto_update = True):
    surfs = [s for s in surfaces if s.vertices is not None]
    cs_class = GradientColor if gradient else VolumeColor
    for surf in surfs:
        cs = cs_class(surf, map, palette, range, transparency = transparency,
                      offset = offset, auto_recolor = auto_update)
        cs.set_vertex_colors()

# -----------------------------------------------------------------------------
#
def _use_full_range(range, palette):
    return (range == 'full'
            or (range is None and (palette is None or not palette.values_specified)))

# -----------------------------------------------------------------------------
#
def _colormap_with_range(cmap, range, default = 'redblue'):
    if cmap is None:
        from chimerax.core.colors import BuiltinColormaps
        cmap = BuiltinColormaps[default]
    if range is None or range[0] is None:
        cm = cmap
    else:
        vmin, vmax = range
        if cmap.values_specified:
            cm = cmap.rescale_range(vmin, vmax)
        else:
            cm = cmap.linear_range(vmin, vmax)
    return cm

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
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
            from .updaters import add_updater_for_session_saving
            add_updater_for_session_saving(surface.session, self)

    # -------------------------------------------------------------------------
    #
    def set_volume(self, volume):

        self.volume = volume

#        from chimera import addModelClosedCallback
#        addModelClosedCallback(volume, self.volume_closed_cb)

    # -------------------------------------------------------------------------
    #
    def set_colormap(self, palette, range, per_pixel = False):
        r = self.value_range() if _use_full_range(range, palette) else range
        self.colormap = _colormap_with_range(palette, r)
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
        rgba = cmap.interpolated_rgba(values)
        if len(outside) > 0:
            from chimerax import map
            map.set_outside_volume_colors(outside, cmap.color_no_value, rgba)

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
        v = _inside_values(values, outside)
        return _array_value_range(v)
        
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
                vo = _offset_vertices(v, n, o)
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
            vo = _offset_vertices(v, n, self.offset)
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
            return None
        vol = data['volume']
        if surf is None:
            session.logger.warning('Could not restore coloring on surface %s because volume does not exist.'
                                   % surf.name)
            return None
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
def _inside_values(values, outside):
        
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
def _array_value_range(a):

    if len(a) == 0:
        return (None, None)
    from numpy import minimum, maximum
    r = (minimum.reduce(a), maximum.reduce(a))
    return r
    
# -----------------------------------------------------------------------------
#
def _offset_vertices(vertices, normals, offset):

    if offset == 0:
        return vertices
    vo = offset * normals
    vo += vertices
    return vo
