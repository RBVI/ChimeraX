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

_color_map_args_doc = '''

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
      Percent transparency to use.  If not specified then palette transparency values are used.
    update : bool
      Whether to automatically update the surface coloring when the surface shape changes.
'''

# -----------------------------------------------------------------------------
#
def color_sample(session, surfaces, map, palette = None, range = None, key = False,
                 offset = 0, transparency = None, update = True, undo_state = None):
    '''
    Color surfaces using an interpolated map value at each surface vertex
    with values mapped to colors by a color palette.
    '''
    _color_by_map_value(session, surfaces, map, palette = palette, range = range, key = key,
                        offset = offset, transparency = transparency, auto_update = update,
                        undo_name = 'color sample', undo_state = undo_state)

color_sample.__doc__ += _color_map_args_doc

# -----------------------------------------------------------------------------
#
def color_electrostatic(session, surfaces, map, palette = None, range = None, key = False,
                        offset = 1.4, transparency = None, update = True):
    '''
    Color surfaces using an interpolated electrostatic potential map value
    at each surface vertex with values mapped to colors by a color palette.
    '''

    if range is None and (palette is None or not palette.values_specified):
        range = (-10,10)
        
    _color_by_map_value(session, surfaces, map, palette = palette, range = range, key = key,
                        offset = offset, transparency = transparency, auto_update = update,
                        undo_name = 'color electrostatic')
    
color_electrostatic.__doc__ += _color_map_args_doc

# -----------------------------------------------------------------------------
#
def color_gradient(session, surfaces, map = None, palette = None, range = None, key = False,
                   offset = 0, transparency = None, update = True):
    '''
    Color surfaces using an map gradient norm value at each surface vertex
    with values mapped to colors by a color palette.
    '''

    if map is None:
        from chimerax.map import VolumeSurface
        if len(surfaces) != 1 or not isinstance(surfaces[0], VolumeSurface):
            from chimerax.core.errors import UserError
            raise UserError('volume gradient command must specify "map" option')
        map = surfaces[0].volume
            
    _color_by_map_value(session, surfaces, map, palette = palette, range = range, key = key,
                        offset = offset, transparency = transparency, gradient = True,
                        auto_update = update, undo_name = 'color gradient')

color_gradient.__doc__ += _color_map_args_doc

# -----------------------------------------------------------------------------
#
def color_surfaces_by_map_value(atoms = None, opacity = None, map = None,
                                palette = None, range = None, offset = 0,
                                undo_state = None):
    from chimerax import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    if len(surfs) == 0:
        return 0
    
    for s in surfs:
        if undo_state:
            cprev = s.color_undo_state
        cs = VolumeColor(s, map, palette, range, offset = offset)
        satoms = s.atoms if atoms is None else atoms
        colored = s.color_atom_patches(satoms, vertex_colors = cs.vertex_colors(), opacity = opacity)
        if undo_state and colored:
            undo_state.add(s, 'color_undo_state', cprev, s.color_undo_state)

    return len(surfs)

# -----------------------------------------------------------------------------
#
def _color_by_map_value(session, surfaces, map, palette = None, range = None, key = False,
                        offset = 0, transparency = None, gradient = False, caps_only = False,
                        auto_update = True, undo_name = 'color map by value', undo_state = None):

    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No surface models specified for coloring by map value')

    surfs = [s for s in surfaces if s.vertices is not None]
    if len(surfs) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Only empty surface models specified for coloring by map value')
    
    cs_class = GradientColor if gradient else VolumeColor

    if undo_state is None:
        from chimerax.core.undo import UndoState
        undo = UndoState(undo_name)
    else:
        undo = undo_state

    for surf in surfs:
        cprev = surf.color_undo_state
        cs = cs_class(surf, map, palette, range, transparency = transparency,
                      offset = offset, auto_recolor = auto_update)
        cs.set_vertex_colors()
        undo.add(surf, 'color_undo_state', cprev, surf.color_undo_state)
        if key:
            from chimerax.color_key import show_key
            show_key(session, cs.colormap)

    if undo_state is None:
        session.undo.register(undo)

    return cs

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
        
        arv = self._auto_recolor if auto_recolor else None
        surface.auto_recolor_vertices = arv

        if auto_recolor:
            from .updaters import add_updater_for_session_saving
            add_updater_for_session_saving(surface.session, self)

    # -------------------------------------------------------------------------
    #
    def active(self):
        s = self.surface
        return s is not None and s.auto_recolor_vertices == self._auto_recolor

    # -------------------------------------------------------------------------
    #
    def set_colormap(self, palette, range, per_pixel = False):
        r = self.value_range() if _use_full_range(range, palette) else range
        self.colormap = _colormap_with_range(palette, r)
        self.per_pixel_coloring = per_pixel
        self.set_texture_colormap()
        
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
    def _auto_recolor(self):
        if self.closed():
            # Volume has been deleted.
            self.surface.auto_recolor_vertices = None
            return
        self.set_vertex_colors(report_stats = False)
        
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
        v = s.vertices
        if v is None:
            return None, None
        n = s.normals
        # Transform from surface to scene coordinates
        tf = s.scene_position
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
    def closed(self):
        return self.volume.deleted
    
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
        if vol is None:
            session.logger.warning('Could not restore coloring on surface %s because volume does not exist.'
                                   % surf.name)
            return None
        c = cls(surf, vol, palette = data['colormap'], range = None,
                transparency = data['transparency'], offset = data['offset'])
        c.set_vertex_colors()
        return c

# -----------------------------------------------------------------------------
#
def volume_coloring(surface):
    '''Return VolumeColor class for surface model if it is being auto colored.'''
    arv = surface.auto_recolor_vertices
    if hasattr(arv, '__self__'):
        vc = arv.__self__  # Instance of a bound method
        if isinstance(vc, VolumeColor):
            return vc
    return None

# -----------------------------------------------------------------------------
#
class GradientColor(VolumeColor):

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
            
# -----------------------------------------------------------------------------
#
def _inside_values(values, outside):
        
    if len(outside) == 0:
        return values
    
    n = len(values)
    if len(outside) == n:
        return ()

    from numpy import zeros, array, put, subtract, nonzero, take
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
