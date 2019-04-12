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
# Create a partially transparent 3d image model from volume data and a color map.
#
# Call update_drawing() to display the model with current levels, colors,
# and rendering options.  Argument align can be a model to align with.
#
class ImageRender:

  def __init__(self, name, grid_data, region, colormap, rendering_options,
               session, blend_manager):

    self.name = name
    self._data = grid_data
    self._region = region
    self._last_ijk_to_xyz_transform = grid_data.ijk_to_xyz_transform
    self._colormap = colormap
    self._rendering_options = rendering_options.copy()

    self._blend_manager = blend_manager	# ImageBlendManager to blend colors with other images
    if blend_manager:			# is None for BlendedImage.
      blend_manager.add_image(self)

    self._session = session
    self._drawing = ImageDrawing(session, self)
    self._color_tables = {}			# Maps axis to (ctable, ctable_range)
    self._c_mode = self._auto_color_mode()	# Explicit mode, cannot be "auto".
    self._mod_rgba = self._luminance_color()	# For luminance color modes.
    self._multiaxis_planes = [None, None, None]	# For x, y, z axis projection
    self._planes_drawing = None			# For ortho and box mode display
    self._view_aligned_planes = None		# ViewAlignedPlanes instance for 3d projection mode
    
  # ---------------------------------------------------------------------------
  #
  def message(self, text, large_data_only = True):

    if self.message_cb:
      if large_data_only:
        si,sj,sk = self.size
        if si*sj*sk <= (1 << 26):
          return
      self.message_cb(text)

  # ---------------------------------------------------------------------------
  #
  def set_region(self, region):
      if region == self._region and self._data.ijk_to_xyz_transform == self._last_ijk_to_xyz_transform:
        return

      self._last_ijk_to_xyz_transform = self._data.ijk_to_xyz_transform
      
      same_step = (region[2] == self._region[2])
      self._region = region
      if self._rendering_options.full_region_on_gpu and same_step:
        self._update_planes_for_new_region()
      else:
        self._remove_planes()
        if not same_step:
          self._need_color_update()

      bi = self._blend_image
      if bi and self is bi.master_image:
        bi.set_region(region)
        self._drawing.redraw_needed()	# Force redraw since BlendedImage is not in draw hierarchy.

  # ---------------------------------------------------------------------------
  #
  def _update_planes_for_new_region(self):
    for d in self._planes_drawings:
      d._update_region = True
      d.redraw_needed()

  # ---------------------------------------------------------------------------
  #
  def set_colormap(self, colormap):
      if colormap != self._colormap:
          self._colormap = colormap
          self._need_color_update()
      
  # ---------------------------------------------------------------------------
  #
  def _need_color_update(self):
    self._color_tables.clear()
    self._c_mode = self._auto_color_mode()
    self._mod_rgba = self._luminance_color()
    mc = self._modulation_color
    for d in self._planes_drawings:
      d._update_colors = True
      d.color = mc
    self._drawing.redraw_needed()
    bi = self._blend_image
    if bi:
      bi._need_color_update()
      
  # ---------------------------------------------------------------------------
  #
  def set_options(self, rendering_options):
# TODO: Detect changes in grid data ijk_to_xyz_transform, matrix value changes...

    # TODO: Don't delete textures unless attribute changes require it, eg. plane_spacing, linear_interpolation
    ro = self._rendering_options
    self._rendering_options = rendering_options.copy()

    change = False
    for attr in ('color_mode', 'colormap_on_gpu', 'colormap_size', 'dim_transparent_voxels',
                 'projection_mode', 'plane_spacing', 'full_region_on_gpu',
                 'orthoplanes_shown', 'orthoplane_positions', 'box_faces', 'linear_interpolation'):
        if getattr(rendering_options, attr) != getattr(ro, attr):
            change = True
            break
    if change:
        self._remove_planes()
        self._need_color_update()
# TODO: Should not delete all textures when one orthoplane is moved.
# TODO: Don't have opengl context current, needed when blending on gpu when orthoplane changes.
#    elif rendering_options.orthoplane_positions != ro.orthoplane_positions:
#      self._update_planes_for_new_region()

    if rendering_options.maximum_intensity_projection != ro.maximum_intensity_projection:
      self._drawing.redraw_needed()	# MIP blending is entirely handled in draw routine.
      
# TODO: _p_mode not used.  Why?
    self._p_mode = self._auto_projection_mode()

    bi = self._blend_image
    if bi and self is bi.master_image:
      bi.set_options(rendering_options)

  @property
  def _ijk_to_xyz(self):
      return self._data.ijk_to_xyz_transform

  @property
  def _single_plane(self):
      for s in self._region_size:
          if s == 1:
              return True
      return False

  @property
  def _showing_view_aligned(self):
    ro = self._rendering_options
    return (ro.projection_mode == '3d'
            and not self._single_plane
            and not ro.any_orthoplanes_shown()
            and not ro.box_faces)
  
  @property
  def _modulation_color(self):
    return tuple(int(255*r) for r in self._mod_rgba)

  @property
  def _opaque(self):
    return (not 'a' in self._c_mode)

  @property
  def _blend_image(self):
    b = self._blend_manager
    return b.blend_image(self) if b else None

  def _update_blend_groups(self):
    b = self._blend_manager
    if b:
      b.update_groups()
    
  # ---------------------------------------------------------------------------
  #
  def _matrix_plane(self, k, axis):
    ijk_min, ijk_max, ijk_step = self._texture_region
    ijk_origin = list(ijk_min)
    ijk_origin[axis] = k
    ijk_size = [i1-i0+1 for i0,i1 in zip(ijk_min, ijk_max)]
    ijk_size[axis] = 1
    m = self._data.matrix(ijk_origin, ijk_size, ijk_step)
    from numpy import squeeze
    p = squeeze(m, 2-axis)	# Reduce from 3d array to 2d.
    return p
      
  # ---------------------------------------------------------------------------
  #
  @property
  def _use_gpu_colormap(self):
    return self._rendering_options.colormap_on_gpu
  @property
  def _use_gpu_blending(self):
    return self._rendering_options.blend_on_gpu

  # ---------------------------------------------------------------------------
  #
  def _color_plane(self, plane, axis, view_aligned=False, require_color=False):

    m = self._matrix_plane(plane, axis)
    if self._rendering_options.colormap_on_gpu and not require_color:
      return m

    cmap, cmap_range = self._color_table() if view_aligned else self._color_table(axis)
    dmin, dmax = cmap_range

    colors = self._color_array(cmap.dtype, tuple(m.shape) + (cmap.shape[1],))
    from . import _map
    _map.data_to_colors(m, dmin, dmax, cmap, self._colormap.clamp, colors)

    if hasattr(self, 'mask_colors'):
      s = [slice(None), slice(None), slice(None)]
      s[2-axis] = plane
      self.mask_colors(colors, slice = s)

    return colors

  # ---------------------------------------------------------------------------
  # Reuse current volume color array if it has correct size.
  # This gives 2x speed-up over allocating a new array when flipping
  # through planes.
  #
  def _color_array(self, ctype, cshape):

    v = self._drawing
    if hasattr(v, '_grayscale_color_array'):
      colors = v._grayscale_color_array
      if colors.dtype == ctype and tuple(colors.shape) == cshape:
        return colors

    from numpy import empty
    try:
      colors = empty(cshape, ctype)
    except MemoryError:
      self.message("Couldn't allocate color array of size (%d,%d,%d,%d) region" % cshape, large_data_only = False)
      raise
    v._grayscale_color_array = colors        # TODO: make sure this array is freed.
    return colors
  
  # ---------------------------------------------------------------------------
  # Returned values are uint8 or uint16 with 4 (RGBA), 3 (RGB), 2 (LA), or 1 (L)
  # components per color depending on color mode.
  # Transparency and brightness adjustments are applied to transfer function.
  # Color map is for a specific set of planes and depends on plane spacing.
  # If axis is None then colormap for 3d projection is returned.
  #
  def _color_table(self, axis = None):
    ctables = self._color_tables
    if axis in ctables:
      cmap, cmap_range = ctables[axis]
      return cmap, cmap_range

    cmap = self._colormap
    tf = cmap.transfer_function

    if self._c_mode.startswith('l'):
      tf, mc = _luminance_transfer_function(tf)

    size, drange, ctype = self._colormap_properties()
    dmin, dmax = drange

    if len(tf) < 2:
      nc = len(self._colormap_components())
      from numpy import zeros
      icmap = zeros((size,nc), ctype)
      ctables[axis] = icmap, drange
      return icmap, drange

    # Convert transfer function to a colormap.
    from numpy import zeros, float32, array
    tfcmap = zeros((size,4), float32)
    tfa = array(tf, float32)
    from ._map import transfer_function_colormap
    transfer_function_colormap(tfa, dmin, dmax, tfcmap)

    # Adjust brightness of RGB components.
    bf = cmap.brightness_factor
    if not self._rendering_options.dim_transparent_voxels:
      # Reduce brightness for closer spaced planes
      # so brightness per unit thickness stays the same.
      bf *= (self._plane_spacing(axis)/max(self._plane_spacings()))
    tfcmap[:,:3] *= bf

    # Modify colormap transparency.
    if cmap.transparency_thickness is not None:
      planes = cmap.transparency_thickness / self._plane_spacing(axis)
      alpha = tfcmap[:,3]
      if planes == 0:
        alpha[:] = 1
      else:
        # TODO: In order to handle non-isotropic grid spacing, the colormap
        # transparency should depend on whether x, y, or z planes are used.
        # Currently there is no dependency on the plane axis. Larger plane
        # spacing should result it increased opacity.  This could be achieved
        # by multiplying the planes value by (min plane spacing / plane spacing).
        # If 3d texture rendering is used, the transparency should depend on
        # the plane spacing being used.
        trans = (alpha < 1)         # Avoid divide by zero for alpha == 1.
        atrans = alpha[trans]
        alpha[trans] = 1.0 - (1.0-atrans) ** (1.0/planes)

    # Use only needed color components (e.g. rgba, la, l).
    cmap = self._rgba_to_colormap(tfcmap)

    # Convert from float to uint8 or uint16.
    from numpy import empty
    icmap = empty(cmap.shape, ctype)
    from . import _map
    _map.colors_float_to_uint(cmap, icmap)

    ctables[axis] = icmap, drange

    return icmap, drange

  # ---------------------------------------------------------------------------
  #
  def _plane_spacing(self, axis = None):
    # Use view aligned spacing equal to maximum grid spacing along 3 axes.
    # This gives highest rendering speed.  Using minimum grid spacing may
    # give higher quality appearance.
    spacings = _plane_spacings(self._region, self._ijk_to_xyz)
    if axis is None:
      ps = self._rendering_options.plane_spacing
      if ps == 'max':
        s = max(spacings)
      elif ps == 'min':
        s = min(spacings)
      elif ps == 'mean':
        from numpy import mean
        s = mean(spacings)
      else:
        s = ps
    else:
      s = spacings[axis]
    return s

  # ---------------------------------------------------------------------------
  #
  def _plane_spacings(self):
    '''Spacing of displayed planes.'''
    return _plane_spacings(self._region, self._ijk_to_xyz)

  # ---------------------------------------------------------------------------
  #
  @property
  def _plane_size(self):
    return self._region_size[:2]
  @property
  def _region_size(self):
    ijk_min, ijk_max, ijk_step = self._region
    return tuple(i1//s - i0//s + 1 for i0,i1,s in zip(ijk_min, ijk_max, ijk_step))
  
  # ---------------------------------------------------------------------------
  #
  def _colormap_properties(self):

    # Color component type
    from numpy import uint8, int8, uint16, int16
    m = self._c_mode
    if m.endswith('8') or m.endswith('4'):      t = uint8
    elif m.endswith('16') or m.endswith('12'):  t = uint16
    else:                                       t = uint8

    # If data is 8-bit or 16-bit integer (signed or unsigned) then use data
    # full type range for colormap so data can be used as colormap index.
    dtype = self._data.value_type.type
    if dtype in (uint8, int8, uint16, int16) and not self._use_gpu_colormap:
      drange = dmin, dmax = _value_type_range(dtype)
      size = (dmax - dmin + 1)
      return size, drange, t

    ro = self._rendering_options
    size = min(ro.colormap_size, 2 ** 16)

    tf = self._colormap.transfer_function
    n = len(tf)
    if n >= 2:
      drange = tf[0][0], tf[n-1][0]
    elif n == 1:
      drange = tf[0][0], tf[0][0]
    else:
      drange = 0,0

    return size, drange, t

  # ---------------------------------------------------------------------------
  # Convert rgba colormap to format appropriate for color mode (e.g. la).
  #
  def _rgba_to_colormap(self, colormap):

    c = self._colormap_components()
    from numpy import empty
    cmap = empty((colormap.shape[0],len(c)), colormap.dtype)
    for i,ci in enumerate(c):
      cmap[:,i] = colormap[:,ci]
    return cmap

  # ---------------------------------------------------------------------------
  # Tuple of colormap component numbers 0=R, 1=G, 2=B, 3=A for mapping RGBA
  # to a format appropriate for color mode.
  #
  def _colormap_components(self):

    m = self._c_mode
    if m.startswith('rgba'):    c = (0,1,2,3)  # RGBA
    elif m.startswith('rgb'):   c = (0,1,2)    # RGB
    elif m.startswith('la'):    c = (0,3)      # RA
    elif m.startswith('l'):     c = (0,)       # R
    else:                       c = (0,1,2,3)  # RGBA
    return c

  # ---------------------------------------------------------------------------
  # 
  def _auto_color_mode(self):

    cm = self._rendering_options.color_mode
    auto = cm.startswith('auto')
    opaque = cm.startswith('opaque')
    if auto or opaque:
      cmap = self._colormap
      from numpy import array
      tf = array(cmap.transfer_function)
      if len(tf) == 0 or hasattr(self, 'mask_colors'):
        m = 'rgb' if opaque else 'rgba'
      else:
        single_color = _colinear(tf[:,2:5], 0.99)
        m = 'l' if single_color else 'rgb'
        if not opaque:
          if cmap.transparency_thickness != 0:
            tfopaque = ((tf[:,5] == 1).all() and (tf[:,1] == 1).all())
            if not tfopaque:
              m += 'a'
      bits = cm[4:] if auto else cm[6:]
      m += bits            # Append bit count
    else:
      m = cm
    return m

  # ---------------------------------------------------------------------------
  #
  def _luminance_color(self):

    if self._c_mode.startswith('l'):
      ltf, rgba = _luminance_transfer_function(self._colormap.transfer_function)
    else:
      rgba = (1,1,1,1)
    return rgba

  # ---------------------------------------------------------------------------
  # 
  def _auto_projection_mode(self):

    pm = self._rendering_options.projection_mode
    if pm == 'auto':
      s = [n*sp for n,sp in zip(self._region_size, self._plane_spacings())]
      smin, smid = sorted(s)[:2]
      aspect_cutoff = 4
      if smin > 0 and aspect_cutoff*smin <= smid:
        pm = ('2d-x','2d-y','2d-z')[list(s).index(smin)]
      else:
        pm = '2d-xyz'
    return pm

  # ---------------------------------------------------------------------------
  #
  def model(self):

    return self._drawing
    
  # ---------------------------------------------------------------------------
  #
  def close_model(self):

    self._remove_planes()
    v = self._drawing
    if v and not v.deleted and v.parent:
      v.parent.remove_drawing(v)
    self._drawing = None

  def _update_planes(self, renderer):
    # Create or update the planes.
    view_dir = self._view_direction(renderer)
    if self._showing_view_aligned:
      self._remove_axis_planes()
      pd = self._update_view_aligned_planes(view_dir)
    else:
      self._remove_view_planes()
      pd = self._update_axis_aligned_planes(view_dir)
    return pd

  def _update_axis_aligned_planes(self, view_direction):
    # Render grid aligned planes
    axis, rev = self._projection_axis(view_direction)
    pd = self._axis_planes(axis)

    if axis is not None:
      # Reverse drawing order if needed to draw back to front
      pd.multitexture_reverse_order = rev
      sc = self._drawing.shape_changed
      for d in self._multiaxis_planes:
        disp = (d is pd)
        if d and d.display != disp:
          # TODO: Make drawing not cause redraw if display value does not change.
          d.display = disp
      # When switching planes, do not set shape change flag
      # since that causes center of rotation update with
      # front center rotation method, which messes up spin movies.
      self._drawing.shape_changed = sc

    return pd

  def _axis_planes(self, axis):
    pd = self._planes_drawing if axis is None else self._multiaxis_planes[axis]
    if pd:
      return pd
    
    sc = self._drawing.shape_changed
    pd = self._make_planes(axis)
    pd._update_colors = self._use_gpu_colormap
    if axis is None:
      self._planes_drawing = pd
    else:
      if tuple(self._multiaxis_planes) != (None, None, None):
        # Reset shape change flag since this is the same shape.
        self._drawing.shape_changed = sc
      self._multiaxis_planes[axis] = pd

    return pd
  
  def _update_view_aligned_planes(self, view_direction):
    pd = self._view_planes()
    pd.update_geometry(view_direction, self._drawing.scene_position)
    return pd

  def _view_planes(self):
    pd = self._view_aligned_planes
    if pd is None:
      ro = self._rendering_options
      pd = ViewAlignedPlanes(self)
      self._view_aligned_planes = pd
      self._drawing.add_drawing(pd)
    return pd

  @property
  def _texture_region(self):
    ro = self._rendering_options
    if ro.full_region_on_gpu:
      ijk_step = self._region[2]
      from .volume import full_region
      tex_region = full_region(self._data.size, ijk_step)
    else:
      tex_region = self._region
    return tex_region

  @property
  def _planes_drawings(self):
    drawings = self._multiaxis_planes + [self._view_aligned_planes, self._planes_drawing]
    return [d for d in drawings if d]
    
  def _remove_planes(self):
    self._remove_axis_planes()
    self._remove_view_planes()
    d = self._drawing
    if d:
      d.redraw_needed()

  def _remove_axis_planes(self):
    pd = self._planes_drawing
    if pd:
      pd.close()
      self._planes_drawing = None

    for pd in self._multiaxis_planes:
      if pd:
        pd.close()
    self._multiaxis_planes = [None,None,None]

  def _remove_view_planes(self):
    pd = self._view_aligned_planes
    if pd:
      pd.close()
      self._view_aligned_planes = None

  def _view_direction(self, render):
    return -render.current_view_matrix.inverse().z_axis()	# View direction, scene coords

  def _projection_axis(self, view_direction):
    # View matrix maps scene to camera coordinates.
    v = view_direction
    ro = self._rendering_options
    if ro.box_faces or ro.any_orthoplanes_shown() or self._showing_view_aligned:
      return None, False

    # Determine which axis has box planes with largest projected area.
    ijk_to_scene = self._drawing.scene_position * self._ijk_to_xyz
    bx,by,bz = ijk_to_scene.axes()	# Box axes, scene coordinates
    # Scale axes to length of box so that plane axis chosen maximizes plane view area for box.
    ijk_min, ijk_max = self._region[:2]
    gs = [i1-i0+1 for i0,i1 in zip(ijk_min, ijk_max)]
    bx *= gs[0]
    by *= gs[1]
    bz *= gs[2]
    from chimerax.core.geometry import cross_product, inner_product
    box_face_normals = [cross_product(by,bz), cross_product(bz,bx), cross_product(bx,by)]
    pmode = ro.projection_mode
    if pmode == '2d-xyz' or pmode == '3d':
      view_areas = [inner_product(v,bfn) for bfn in box_face_normals]
      from numpy import argmax, abs
      axis = argmax(abs(view_areas))
      rev = (view_areas[axis] > 0)
    else:
      axis = {'2d-x': 0, '2d-y': 1, '2d-z': 2}.get(pmode, 2)
      rev = (inner_product(v,box_face_normals[axis]) > 0)

    return axis, rev

  def _make_planes(self, axis):
    d = AxisAlignedPlanes(self, axis)
    self._drawing.add_drawing(d)
    return d

class Colormap:
    def __init__(self, transfer_function,
                 brightness_factor, transparency_thickness, clamp = False):

      self.transfer_function = transfer_function
      self.brightness_factor = brightness_factor
      self.transparency_thickness = transparency_thickness
      self.clamp = clamp

    def __eq__(self, cmap):
        return (self.transfer_function == cmap.transfer_function and
                self.brightness_factor == cmap.brightness_factor and
                self.transparency_thickness == cmap.transparency_thickness and
                self.clamp == cmap.clamp)


from chimerax.core.models import Model
class ImageDrawing(Model):
  SESSION_SAVE = False		# Volume restores this model.

  def __init__(self, session, image_render):
    self._image_render = image_render
    Model.__init__(self, image_render.name, session)

  def delete(self):
    ir = self._image_render
    b = ir._blend_manager
    if b:
      b.remove_image(ir)
    ir._remove_planes()
    Model.delete(self)

  def bounds(self):
    # Override bounds because GrayScaleDrawing does not set geometry until draw() is called
    # but setting initial camera view uses bounds before draw() is called.

    if not self.display:
      return None

    ir = self._image_render
    corners = _box_corners(ir._region, ir._ijk_to_xyz)
    positions = self.get_scene_positions(displayed_only = True)
    from chimerax.core.geometry import point_bounds
    b = point_bounds(corners, positions)
    return b
    
  def drawings_for_each_pass(self, pass_drawings):
    '''Override Drawing method because geometry is not set until draw() is called.'''
    if not self.display:
      return

    opaque = self._image_render._opaque
    p = self.OPAQUE_DRAW_PASS if opaque else self.TRANSPARENT_DRAW_PASS
    if p in pass_drawings:
      pass_drawings[p].append(self)
    else:
      pass_drawings[p] = [self]

    # Do not include child drawings since this drawing overrides draw() method
    # and draws the children.

  def draw(self, renderer, draw_pass):
    if not self.display:
      return

    ir = self._image_render
    ir._update_blend_groups()
    bi = ir._blend_image
    if bi:
      if ir is bi.master_image:
        bi.draw(renderer, draw_pass)
      return

    transparent = not ir._opaque
    from chimerax.core.graphics import Drawing
    dopaq = (draw_pass == Drawing.OPAQUE_DRAW_PASS and not transparent)
    dtransp = (draw_pass == Drawing.TRANSPARENT_DRAW_PASS and transparent)
    if not dopaq and not dtransp:
      return

    pd = ir._update_planes(renderer)

    if pd._update_region:
      pd.update_region()
      pd._update_region = False

    pd._update_coloring()

    self._draw_planes(renderer, draw_pass, dtransp, pd)

  def _draw_planes(self, renderer, draw_pass, dtransp, drawing):
    r = renderer
    ro = self._image_render._rendering_options
    max_proj = dtransp and ro.maximum_intensity_projection
    if max_proj:
      r.blend_max(True)
    if dtransp:
      r.write_depth(False)
    blend1 = (dtransp and not ro.dim_transparent_voxels)
    if blend1:
      r.blend_alpha(False)

    drawing.draw(r, draw_pass)

    if blend1:
      r.blend_alpha(True)
    if dtransp:
      r.write_depth(True)
    if max_proj:
      r.blend_max(False)
  
# ---------------------------------------------------------------------------
#
from chimerax.core.graphics import Drawing
class PlanesDrawing(Drawing):
  def __init__(self, name, image_render, axis = None):
    Drawing.__init__(self, name)
    self._update_region = False
    self._update_colors = True
    self._axis = axis
    self._image_render = image_render
    
  # ---------------------------------------------------------------------------
  #
  def _update_coloring(self):
    if not self._update_colors:
      return

    ir = self._image_render
    if ir._rendering_options.colormap_on_gpu:
      self._update_colormap_texture()
    else:
      self._fill_textures()
    self._update_colors = False

  # ---------------------------------------------------------------------------
  #
  def _update_colormap_texture(self):
    ir = self._image_render
    cmap, cmap_range = ir._color_table(self._axis)
    t = self.colormap
    if t is None:
      from chimerax.core.graphics import Texture
      self.colormap = Texture(cmap, dimension = 1)
    else:
      t.reload_texture(cmap)
    self.colormap_range = cmap_range

  # ---------------------------------------------------------------------------
  #
  def _fill_textures(self):
    '''Must be defined by derived classes.'''
    pass
  
# ---------------------------------------------------------------------------
#
class AxisAlignedPlanes(PlanesDrawing):
  # axis is None for box faces or orthoplanes.
  def __init__(self, image_render, axis):
    name = 'grayscale axis aligned planes'
    PlanesDrawing.__init__(self, name, image_render, axis)

    self.color = image_render._modulation_color
    self.use_lighting = False
    self.opaque_texture = image_render._opaque

    self._textures = {}

    planes = self._set_planes(axis)
    self._update_textures(planes)

  def update_region(self):
    planes = self._set_planes(self._axis)
    self._update_textures(planes)
    
  def _set_planes(self, axis):
    ro = self._image_render._rendering_options
    if axis is not None:
      planes = self._set_axis_planes(axis)
    elif ro.box_faces:
      planes = self._set_box_faces()
    elif ro.any_orthoplanes_shown():
      planes = self._set_ortho_planes()
    return planes
    
  def _set_axis_planes(self, axis):
    ijk_min, ijk_max, ijk_step = self._image_render._region
    planes = tuple((k, axis) for k in range(ijk_min[axis],ijk_max[axis]+1,ijk_step[axis]))
    self._update_geometry(planes)
    return planes

  def _set_ortho_planes(self):
    ro = self._image_render._rendering_options
    p = ro.orthoplane_positions
    show_axis = ro.orthoplanes_shown
    planes = tuple((p[axis], axis) for axis in (0,1,2) if show_axis[axis])
    self._update_geometry(planes)
    return planes
  
  def _set_box_faces(self):
    ijk_min, ijk_max, ijk_step = self._image_render._region
    planes = (tuple((ijk_min[axis],axis) for axis in (0,1,2)) +
              tuple((ijk_max[axis],axis) for axis in (0,1,2)))
    self._update_geometry(planes)
    return planes
  
  def _update_geometry(self, planes):
    ir = self._image_render
    (i0,j0,k0), (i1,j1,k1) = ir._region[:2]
    (fi0,fj0,fk0), (fi1,fj1,fk1) = ir._texture_region[:2]
    from numpy import array, float32, int32, empty
    vaa = array((((0,j0,k0),(0,j1,k0),(0,j1,k1),(0,j0,k1)),
                 ((i0,0,k0),(i1,0,k0),(i1,0,k1),(i0,0,k1)),
                 ((i0,j0,0),(i1,j0,0),(i1,j1,0),(i0,j1,0))), float32)
    tap = array(((0,1,2),(0,2,3)), int32)
    di,dj,dk = max(1, fi1-fi0), max(1, fj1-fj0), max(1, fk1-fk0)
    # Texture coords range from 1/2N to 1-1/2N, not from 0 to 1.
    ti0,ti1,tj0,tj1,tk0,tk1 = (i0-fi0+0.5)/di, (i1-fi0-0.5)/di, (j0-fj0+0.5)/dj, (j1-fj0-0.5)/dj, (k0-fk0+0.5)/dk, (k1-fk0-0.5)/dk
    tca = array((((tj0,tk0),(tj1,tk0),(tj1,tk1),(tj0,tk1)),
                 ((ti0,tk0),(ti1,tk0),(ti1,tk1),(ti0,tk1)),
                 ((ti0,tj0),(ti1,tj0),(ti1,tj1),(ti0,tj1))),
                float32)
    np = len(planes)
    va = empty((4*np,3), float32)
    tc = empty((4*np,2), float32)
    ta = empty((2*np,3), int32)
    axes = set()
    for p, (k,axis) in enumerate(planes):
      vap = va[4*p:4*(p+1),:]
      vap[:,:] = vaa[axis]
      vap[:,axis] = k
      ir._ijk_to_xyz.transform_points(vap, in_place = True)
      ta[2*p:2*(p+1),:] = tap + 4*p
      tc[4*p:4*(p+1),:] = tca[axis]
      axes.add(axis)

    self.set_geometry(va, None, ta)
    self.texture_coordinates = tc
    self._axis = None if len(axes) != 1 else axes.pop()

  def _update_textures(self, planes):
    textures = [self._plane_texture(k, axis) for k,axis in planes]
    self.multitexture = textures
    
  def _plane_texture(self, k, axis):
    t = self._textures.get((k,axis))
    if t is None:
      t = self._make_plane_texture()
      self._fill_plane_texture(k, axis, t)
      self._textures[(k,axis)] = t
    return t

  def _make_plane_texture(self):
    ir = self._image_render
    lo = ir._rendering_options.linear_interpolation
    from chimerax.core.graphics import Texture
    t = Texture(linear_interpolation = lo)
    if isinstance(ir, BlendedImage):
      t.initialize_rgba(ir._plane_size)
    return t

  def _fill_plane_texture(self, plane, axis, texture):
    ir = self._image_render
    if ir._use_gpu_blending and isinstance(ir, BlendedImage):
      self._fill_plane_texture_blended(plane, axis, texture)
    else:
      # Single unblended image.
      data = ir._color_plane(plane, axis)
      texture.reload_texture(data, now = True)

  def _fill_plane_texture_blended(self, plane, axis, texture):
    # Blend textures on GPU

    # First make sure all source image textures are up to date.
    ir = self._image_render
    ro = ir._rendering_options
    iaxis = None if ro.box_faces or ro.any_orthoplanes_shown() else axis
    for si in ir.images:
      ap = si._axis_planes(iaxis)
      ap._update_coloring()	# TODO: Should do this once for all planes.
      ap._plane_texture(plane, axis)

    # Blend in each source texture to produce blended texture.
    b = ir._session.main_view.render.blend
    b.start_blending(texture)
    for si in ir.images:
      ap = si._axis_planes(iaxis)
      src_tex = ap._plane_texture(plane, axis)
      modulation_color = [c/255 for c in si._modulation_color]
      if si._use_gpu_colormap:
        b.blend(src_tex, modulation_color, ap.colormap, ap.colormap_range)
      else:
        b.blend(src_tex, modulation_color)
    b.finish_blending()

  def _fill_textures(self):
    for (k,axis),t in self._textures.items():
      self._fill_plane_texture(k, axis, t)

  def close(self):
    tex = self._textures
    if tex:
      # Drawing may not make opengl context current because it was never drawn
      # if it just was used for blending. So make sure context is current here.
      # TODO: Need more general way to assure textures always deleted with context current.
      r = self._image_render._session.main_view.render
      r.make_current()
      for t in tex.values():
        t.delete_texture()
      tex.clear()
      self.multitexture = None
    self.parent.remove_drawing(self)

# ---------------------------------------------------------------------------
#
class ViewAlignedPlanes(PlanesDrawing):
  
  def __init__(self, image_render):

    name = 'grayscale view aligned planes'
    PlanesDrawing.__init__(self, name, image_render)

    ir = image_render
    self.color = ir._modulation_color
    self.use_lighting = False
    self.opaque_texture = ir._opaque

    self._corners = _box_corners(ir._region, ir._ijk_to_xyz)	# in volume coords
    self._vertex_to_texcoord = _xyz_to_texcoord(ir._texture_region, ir._ijk_to_xyz)
    self._last_view_direction = None

    self._fill_textures()

  def update_region(self):
    ir = self._image_render
    self._corners = _box_corners(ir._region, ir._ijk_to_xyz)
    self._last_view_direction = None
    
  def update_geometry(self, view_direction, scene_position):
    tvd = tuple(view_direction)
    if tvd == self._last_view_direction:
      return
    self._last_view_direction = tvd
    
    va, tc, ta = self._perp_planes_geometry(view_direction, scene_position)
    self.set_geometry(va, None, ta)
    self.texture_coordinates = tc

  def _perp_planes_geometry(self, view_direction, scene_position):

    if scene_position.is_identity():
      axis = -view_direction
    else:
      axis = -scene_position.transpose().transform_vector(view_direction)

    # Find number of cut planes
    corners = self._corners
    from . import offset_range
    omin, omax = offset_range(corners, axis)
    spacing = self._image_render._plane_spacing()
    from math import floor, fmod
    n = int(floor((omax - omin) / spacing))
    
    # Reduce Moire patterns as volume rotated by making a cut plane always intercept box center.
    omid = 0.5*(omin + omax)
    offset = omin + fmod(omid - omin, spacing)

    # Triangulate planes intersecting with volume box
    from . import box_cuts
    va, ta = box_cuts(corners, axis, offset, spacing, n)
    tc = self._vertex_to_texcoord * va
    
    return va, tc, ta

  def _fill_textures(self):
    t = self.texture
    if t is None:
      self.texture = t = self._texture_3d()
      
    if isinstance(self._image_render, BlendedImage):
      self._fill_texture_blend(t)
    else:
      td = self._texture_3d_data()
      t.reload_texture(td, now = True)

  def _texture_3d(self):
    td = self._texture_3d_data()
    ir = self._image_render
    lo = ir._rendering_options.linear_interpolation
    from chimerax.core.graphics import Texture
    t = Texture(td, dimension = 3, linear_interpolation = lo)
    if isinstance(ir, BlendedImage):
      t.initialize_rgba(ir._region_size)
    return t

  def _texture_3d_data(self):
    z_axis = 2
    ir = self._image_render
    ijk_min, ijk_max, ijk_step = ir._texture_region
    k0,k1,kstep = ijk_min[z_axis], ijk_max[z_axis], ijk_step[z_axis]
    k0,k1 = kstep*(k0//kstep), kstep*(k1//kstep)
    p = ir._color_plane(k0, z_axis, view_aligned=True)
    sz = (k1 - k0 + kstep)//kstep
    if sz == 1:
      td = p	# Single z plane
    else:
      from numpy import empty
      td = empty((sz,) + tuple(p.shape), p.dtype)
      td[0,:] = p
      for i in range(1,sz):
        td[i,:] = ir._color_plane(k0+i*kstep, z_axis, view_aligned=True)
    return td

  def _fill_texture_blend(self, texture):
    # Blend textures on GPU

    # First make sure all source image textures are up to date.
    ir = self._image_render
    for si in ir.images:
      si._view_planes()._update_coloring()

    # Blend in each source texture to produce blended texture.
    b = ir._session.main_view.render.blend
    b.start_blending(self.texture)
    for si in ir.images:
      vap = si._view_aligned_planes
      modulation_color = [c/255 for c in si._modulation_color]
      if si._use_gpu_colormap:
        b.blend3d(vap.texture, modulation_color, self.texture, vap.colormap, vap.colormap_range)
      else:
        b.blend3d(vap.texture, modulation_color, self.texture)
    b.finish_blending()

  def close(self):
    tex = self.texture
    if tex:
      # Drawing may not make opengl context current because it was never drawn
      # if it just was used for blending. So make sure context is current here.
      # TODO: Need more general way to assure textures always deleted with context current.
      r = self._image_render._session.main_view.render
      r.make_current()
      tex.delete_texture()
      self.texture = None
    self.parent.remove_drawing(self)

def _box_corners(ijk_region, ijk_to_xyz = None):
    (i0,j0,k0), (i1,j1,k1) = ijk_region[:2]
    # Corner order matters for 3d texture rendering.
    from numpy import array, float32
    corners = array(((i0,j0,k0), (i1,j0,k0), (i0,j1,k0), (i1,j1,k0),
                     (i0,j0,k1), (i1,j0,k1), (i0,j1,k1), (i1,j1,k1)), float32)
    if ijk_to_xyz is None:
        return corners
    xyz_corners = ijk_to_xyz * corners
    return xyz_corners

def _plane_spacings(ijk_region, ijk_to_xyz):
  ijk_step = ijk_region[2]
  return [sp*st for sp,st in zip(ijk_to_xyz.axes_lengths(), ijk_step)]

def _xyz_to_texcoord(ijk_region, ijk_to_xyz):
  # Use texture coord range [1/2n,1-1/2n], not [0,1].
  ijk_min, ijk_max, ijk_step = ijk_region
  ei,ej,ek = [i1-i0+1 for i0,i1 in zip(ijk_min, ijk_max)]
  i0,j0,k0 = ijk_min
  from chimerax.core.geometry.place import scale, translation
  # Map i0 to texture coord 0.5/ei and i1 to (ei-0.5)/ei, the texel centers.
  v_to_tc = scale((1/ei, 1/ej, 1/ek)) * translation((0.5-i0,0.5-j0,0.5-k0)) * ijk_to_xyz.inverse()
  return v_to_tc

# ---------------------------------------------------------------------------
#
class BlendedImage(ImageRender):

  def __init__(self, images):

    name = 'blend ' + ', '.join(ir.name for ir in images)
    i0 = images[0]
    ImageRender.__init__(self, name, i0._data, i0._region,
                         i0._colormap, i0._rendering_options,
                         i0.model().session, blend_manager = None)

    self.images = images

    ro = self._rendering_options
    ro.colormap_on_gpu = False

#    for ir in images:
#      ir._remove_planes()	# Free textures and opengl buffers

    self._rgba8_array = None

  def set_options(self, rendering_options):
    ro = rendering_options.copy()
    ro.colormap_on_gpu = False
    ImageRender.set_options(self, ro)

  def draw(self, renderer, draw_pass):
    self._drawing.draw(renderer, draw_pass)

  @property
  def master_image(self):
    return self.images[0]
      
  def _color_plane(self, k, axis, view_aligned=False):
    p = None
    for ir in self.images:
      dp = ir._color_plane(k, axis, view_aligned=view_aligned, require_color=True)
      cmode = ir._c_mode
      if p is None:
        h,w = dp.shape[:2]
        p = self._color_array(w,h)
        if cmode == 'rgba8':
          p[:] = dp
        elif cmode == 'rgb8':
          p[:,:,:3] = dp
          p[:,:,3] = 255
        elif cmode == 'la8':
          copy_la_to_rgba(dp, ir._mod_rgba, p)
        elif cmode == 'l8':
          copy_l_to_rgba(dp, ir._mod_rgba, p)
        else:
          raise ValueError('Cannot blend with color mode %s' % cmode)
      else:
        if cmode == 'rgba8':
          blend_rgba(dp, p)
        if cmode == 'rgb8':
          blend_rgb_to_rgba(dp, p)
        elif cmode == 'la8':
          blend_la_to_rgba(dp, ir._mod_rgba, p)
        elif cmode == 'l8':
          blend_l_to_rgba(dp, ir._mod_rgba, p)
    return p

  def _color_array(self, w, h):
    # Reuse same array for faster color updating.
    a = self._rgba8_array
    if a is None or tuple(a.shape) != (h, w, 4):
      from numpy import empty, uint8
      self._rgba8_array = a = empty((h,w,4), uint8)
    return a

  def _auto_color_mode(self):
    opaque = self._rendering_options.color_mode.startswith('opaque')
    return 'rgb8' if opaque else 'rgba8'
    

# ---------------------------------------------------------------------------
#
def copy_la_to_rgba(la_plane, color, rgba_plane):
  h, w = la_plane.shape[:2]
  from . import _map
  _map.copy_la_to_rgba(la_plane.reshape((w*h,2)), color, rgba_plane.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
def blend_la_to_rgba(la_plane, color, rgba_plane):
  h, w = la_plane.shape[:2]
  from . import _map
  _map.blend_la_to_rgba(la_plane.reshape((w*h,2)), color, rgba_plane.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
def copy_l_to_rgba(l_plane, color, rgba_plane):
  h, w = l_plane.shape[:2]
  from . import _map
  _map.copy_l_to_rgba(l_plane.reshape((w*h,)), color, rgba_plane.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
def blend_l_to_rgba(l_plane, color, rgba_plane):
  h, w = l_plane.shape[:2]
  from . import _map
  _map.blend_l_to_rgba(l_plane.reshape((w*h,)), color, rgba_plane.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
def blend_rgb_to_rgba(rgb, rgba):
  h, w = rgba.shape[:2]
  from . import _map
  _map.blend_rgb_to_rgba(rgb.reshape((w*h,3)), rgba.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
def blend_rgba(rgba1, rgba2):
  h, w = rgba1.shape[:2]
  from . import _map
  _map.blend_rgba(rgba1.reshape((w*h,4)), rgba2.reshape((w*h,4)))

# ---------------------------------------------------------------------------
#
class ImageBlendManager:
  def __init__(self):
    self.blend_images = set()
    self._blend_image = {}	# Map ImageRender to BlendedImage
    self.need_group_update = False

  def add_image(self, image_render):
    self._blend_image[image_render] = None
    self.need_group_update = True

  def remove_image(self, image_render):
    dbi = self._blend_image
    bi = dbi.get(image_render)
    if bi:
      for i2 in bi.images:
        dbi[i2] = None
      self.blend_images.discard(bi)
      bi.close_model()
    del dbi[image_render]
    self.need_group_update = True

  def blend_image(self, image_render):
    return self._blend_image.get(image_render, None)

  def update_groups(self):
    if not self.need_group_update:
      return
    self.need_group_update = False

    # TODO: Don't update groups unless drawing changed.
    groups = []
    dbi = self._blend_image
    images = list(dbi.keys())
    images.sort(key = lambda d: d.name)
    aligned = {}
    for ir in images:
      d = ir.model()
      ro = ir._rendering_options
      if d.display and d.parents_displayed and not ro.maximum_intensity_projection:
        sortho = ro.orthoplanes_shown
        orthoplanes = (sortho, tuple(ro.orthoplane_positions)) if sortho else sortho
        # Need to have matching grid size, scene position, grid spacing, box face mode, orthoplane mode
        k = (tuple(tuple(ijk) for ijk in ir._region),
             tuple(d.scene_position.matrix.flat),
             tuple(ir._ijk_to_xyz.matrix.flat),
             ro.box_faces, orthoplanes)
        if k in aligned:
          aligned[k].append(ir)
        else:
          g = [ir]
          groups.append(g)
          aligned[k] = g

    igroup = {}
    for g in groups:
      if len(g) >= 2:
        for d in g:
          igroup[d] = g

    # Remove blend images for groups that no longer exist.
    bi_gone = []
    bis = self.blend_images
    for bi in bis:
      i = bi.images[0]
      if igroup.get(i,None) != bi.images:
        bi_gone.append(bi)
    for bi in bi_gone:
      for i in bi.images:
        dbi[i] = None
      bis.discard(bi)
      bi.close_model()
      
    # Created blend images for new groups
    for g in groups:
      if len(g) >= 2:
        if dbi[g[0]] is None:
          bi = BlendedImage(g)
          bis.add(bi)
          for i in g:
            dbi[i] = bi

# ---------------------------------------------------------------------------
#
def blend_manager(session):
  m = getattr(session, '_image_blend_manager', None)
  if m is None:
    session._image_blend_manager = m = ImageBlendManager()
    def need_group_update(*args, m=m):
      m.need_group_update = True
    session.triggers.add_handler('new frame', need_group_update)
  return m

# -----------------------------------------------------------------------------
#
def _luminance_transfer_function(tf):

  if len(tf) == 0:
    return tf, (1,1,1,1)

  ltf = []
  for v,i,r,g,b,a in tf:
    l = 0.3*r + 0.59*g + 0.11*b
    ltf.append((v,i,l,l,l,a))

  # Normalize to make maximum luminance = 1
  from numpy import argmax
  lmi = argmax([r for v,i,r,g,b,a in ltf])
  lmax = ltf[lmi][2]
  if lmax != 0:
    ltf = [(v,i,r/lmax,g/lmax,b/lmax,a) for v,i,r,g,b,a in ltf]
  lcolor = tuple(tf[lmi][2:5]) + (1,)

  return ltf, lcolor

# -----------------------------------------------------------------------------
#
def _colinear(vlist, tolerance = 0.99):

  from numpy import inner
  vnz = [v for v in vlist if inner(v,v) > 0]
  if len(vnz) <= 1:
    return True
  v0 = vnz[0]
  m0 = inner(v0,v0)
  t2 = tolerance * tolerance
  for v in vnz[1:]:
    m = inner(v,v0)
    m2 = m*m
    if m2 < t2 * m0 * inner(v,v):
      return False
  return True

# -----------------------------------------------------------------------------
#
def _value_type_range(numpy_type):

  from numpy import uint8, int8, uint16, int16
  tsize = {
    uint8: (0, 255),
    int8: (-128, 127),
    uint16: (0, 65535),
    int16: (-32768, 32767),
    }
  return tsize.get(numpy_type, (None, None))
