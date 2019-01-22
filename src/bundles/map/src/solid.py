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
# Create a partially transparent solid model from volume data and a color map.
#
# Call update_drawing() to display the model with current levels, colors,
# and rendering options.  Argument align can be a model to align with.
#
class Solid:

  def __init__(self, name, size, value_type, matrix_id, matrix_plane, transform,
               align = None, message_cb = None):

    self.name = name

    self.size = size			# 3D matrix size
    self.value_type = value_type	# Numpy dtype of matrix values
    self.matrix_id = matrix_id          # indicates if matrix changed
    self.matrix_plane = matrix_plane    # Callback to get matrix values

    self.transform = transform		# Matrix indices to xyz

    self.drawing = None                 # GrayScaleDrawing
    self.add_handler = None

    self.attached_model = align

    self.message_cb = message_cb

    self.transfer_function = ()
    self.brightness_factor = 1
    self.transparency_thickness = 1	# In xyz space units.  Opacity values are for this thickness.
    self.colormap_size = 256
    self.clamp = False
    self._colormaps = {}		# Maps axis to (cmap, cmap_range)

    self.color_mode = 'auto8'
    self.c_mode = 'rgba8'
    self.projection_mode = 'auto'
    self.p_mode = '2d-xyz'
    self.use_plane_callback = True      # Avoids allocating 3d color array
    self.dim_transparent_voxels = True
    self.bt_correction = False
    self.minimal_texture_memory = False
    self.maximum_intensity_projection = False
    self.linear_interpolation = True
    self.show_outline_box = True
    self.outline_box_rgb = (1,1,1)
    self.outline_box_linewidth = 1
    self.box_faces = False
    self.orthoplanes_shown = (False, False, False)
    self.orthoplane_mijk = (0,0,0)

    self._update_colors = False
    
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
  def set_matrix(self, size, value_type, matrix_id, matrix_plane):

    if (matrix_id != self.matrix_id or
        tuple(size) != self.size or
        value_type != self.value_type):
      self.size = tuple(size)
      self.value_type = value_type
      self.matrix_id = matrix_id
      self.matrix_plane = matrix_plane    # Callback to get matrix values
      self.p_mode = self.auto_projection_mode()
      self._update_colors = True

  # ---------------------------------------------------------------------------
  #
  def set_colormap(self, transfer_function,
                   brightness_factor, transparency_thickness, clamp = False):

    if (self.transfer_function != transfer_function or 
        self.brightness_factor != brightness_factor or
        self.transparency_thickness != transparency_thickness or
        self.clamp != clamp):
      self.transfer_function = transfer_function
      self.brightness_factor = brightness_factor
      self.transparency_thickness = transparency_thickness
      self.clamp = clamp
      self.c_mode = self.auto_color_mode()              # update color mode
      self._update_colors = True
      
  # ---------------------------------------------------------------------------
  # After setting options need to call update_drawing() update display.
  #
  def set_options(self, color_mode, projection_mode,
                  dim_transparent_voxels, bt_correction, minimal_texture_memory,
		  maximum_intensity_projection, linear_interpolation,
                  show_outline_box, outline_box_rgb, outline_box_linewidth,
                  box_faces, orthoplanes_shown, orthoplane_mijk):

    if color_mode != self.color_mode:
      self._update_colors = True
    self.color_mode = color_mode
    self.c_mode = self.auto_color_mode()
    self.projection_mode = projection_mode
    self.p_mode = self.auto_projection_mode()
    self.dim_transparent_voxels = dim_transparent_voxels
    self.bt_correction = bt_correction
    self.minimal_texture_memory = minimal_texture_memory
    self.maximum_intensity_projection = maximum_intensity_projection
    self.linear_interpolation = linear_interpolation
    self.show_outline_box = show_outline_box
    self.outline_box_rgb = outline_box_rgb
    self.outline_box_linewidth = outline_box_linewidth
    self.box_faces = box_faces
    self.orthoplanes_shown = orthoplanes_shown
    self.orthoplane_mijk = orthoplane_mijk
    
  # ---------------------------------------------------------------------------
  #
  def set_transform(self, transform):

    self.transform = transform
    
  # ---------------------------------------------------------------------------
  #
  def image_blend(self, b):

    self.blend_volumes = b
    if b:
      if self.c_mode != 'rgba8':
        self.color_mode = self.c_mode = 'rgba8'
        self._update_colors = True
    d = self.drawing
    if d:
      d.image_blend = b

  # ---------------------------------------------------------------------------
  #
  def update_drawing(self, parent_drawing, blend_manager):

    d = self.drawing
    if d is None:
      self.drawing = d = self.make_drawing(parent_drawing, blend_manager)

    d.display = True
    d.set_array_coordinates(self.transform)
    d.set_color_mode(self.c_mode)
    d.set_modulation_rgba(self.luminance_color())
    d.projection_mode = self.p_mode
    if self.dim_transparent_voxels:
      bmode = d.SRC_ALPHA_DST_1_MINUS_ALPHA
    else:
      bmode = d.SRC_1_DST_1_MINUS_ALPHA
    d.transparency_blend_mode = bmode
    d.brightness_and_transparency_correction = self.bt_correction
    d.minimal_texture_memory = self.minimal_texture_memory
    d.maximum_intensity_projection = self.maximum_intensity_projection
    d.set_linear_interpolation(self.linear_interpolation)
    d.show_outline_box = self.show_outline_box
    d.outline_box_rgb = self.outline_box_rgb
    d.outline_box_linewidth = self.outline_box_linewidth
    d.show_box_faces = self.box_faces
    axis_bits = 0
    for a in (0,1,2):
      if self.orthoplanes_shown[a]:
        axis_bits |= (1 << a)
    d.show_ortho_planes = axis_bits
    d.ortho_planes_position = self.orthoplane_mijk
#    d.image_blend = self.blend_volumes

    self._update_coloring()
    
  # ---------------------------------------------------------------------------
  #
  def make_drawing(self, parent_drawing, blend_manager):

    from . import grayscale
    gsd = grayscale.GrayScaleDrawing(self.name, blend_manager)
    parent_drawing.add_drawing(gsd)
    self._update_colors = True

    return gsd
    
  # ---------------------------------------------------------------------------
  #
  def _update_coloring(self):
    if not self._update_colors:
      return
    self._update_colors = False
    
    if self.use_plane_callback:
      self._colormaps.clear()
      d = self.drawing
      d.set_color_plane_callback(self.size, self.color_plane)
      d.set_color_mode(self.c_mode)
    else:
      colors = self.color_3d()
      self.drawing.set_volume_colors(colors)

  # ---------------------------------------------------------------------------
  #
  def color_plane(self, plane, axis, view_aligned=False):

    cmaps = self._colormaps
    av = (axis, view_aligned)
    if av in cmaps:
      cmap, cmap_range = cmaps[av]
    else:
      cmap, cmap_range = self.colormap() if view_aligned else self.colormap(axis)
      cmaps[av] = cmap, cmap_range
    dmin, dmax = cmap_range

    m = self.matrix_plane(axis, plane)

    colors = self.color_array(cmap.dtype, tuple(m.shape) + (cmap.shape[1],))
    from . import _map
    _map.data_to_colors(m, dmin, dmax, cmap, self.clamp, colors)

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
  def color_array(self, ctype, cshape):

    v = self.drawing
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
  def colormap(self, axis = None):

    tf = self.transfer_function

    if self.c_mode.startswith('l'):
      tf, mc = luminance_transfer_function(tf)

    size, drange, ctype = self.colormap_properties()
    dmin, dmax = drange

    if len(tf) < 2:
      nc = len(self.colormap_components())
      from numpy import zeros
      icmap = zeros((size,nc), ctype)
      return icmap, drange

    # Convert transfer function to a colormap.
    from numpy import zeros, float32, array
    tfcmap = zeros((size,4), float32)
    tfa = array(tf, float32)
    from ._map import transfer_function_colormap
    transfer_function_colormap(tfa, dmin, dmax, tfcmap)

    # For 3d projection use smallest plane spacing
    if axis is None:
      plane_spacings = self.transform.axes_lengths()
      from numpy import argmin
      axis = argmin(plane_spacings)
      
    # Adjust brightness of RGB components.
    bf = self.brightness_factor
    if not self.dim_transparent_voxels:
      # Reduce brightness for closer spaced planes
      # so brightness per unit thickness stays the same.
      plane_spacings = self.transform.axes_lengths()
      bf *= (plane_spacings[axis]/max(plane_spacings))
    tfcmap[:,:3] *= bf

    # Modify colormap transparency.
    if self.transparency_thickness is not None:
      plane_spacing = self.transform.axis_length(axis)
      planes = self.transparency_thickness / plane_spacing
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
    cmap = self.rgba_to_colormap(tfcmap)

    # Convert from float to uint8 or uint16.
    from numpy import empty
    icmap = empty(cmap.shape, ctype)
    from . import _map
    _map.colors_float_to_uint(cmap, icmap)

    return icmap, drange

  # ---------------------------------------------------------------------------
  # Colormaps for the 3 axes.  They can differ in opacity due to different
  # plane spacing along each axis.
  #
  def colormaps(self):
    if len(set(self.transform.axes_lengths())) == 1:
      # Same transparency depth on all 3 axes.  Use one colormap.
      cmap, cmap_range = self.colormap()
      cmaps = (cmap, cmap, cmap)
      cmap_ranges = (cmap_range, cmap_range, cmap_range)
    else:
      # Different transparency depths on some axes.  Use multiple colormaps.
      cmaps = []
      cmap_ranges = []
      for axis in (0,1,2):
        cmap, cmap_range = self.colormap(axis=axis)
        cmaps.append(cmap)
        cmap_ranges.append(cmap_range)
    return cmaps, cmap_ranges

  # ---------------------------------------------------------------------------
  #
  def colormap_properties(self):

    # Color component type
    from numpy import uint8, int8, uint16, int16
    m = self.c_mode
    if m.endswith('8') or m.endswith('4'):      t = uint8
    elif m.endswith('16') or m.endswith('12'):  t = uint16
    else:                                       t = uint8

    # If data is 8-bit or 16-bit integer (signed or unsigned) then use data
    # full type range for colormap so data can be used as colormap index.
    dtype = self.value_type.type
    if dtype in (uint8, int8, uint16, int16):
      drange = dmin, dmax = value_type_range(dtype)
      size = (dmax - dmin + 1)
      return size, drange, t

    size = min(self.colormap_size, 2 ** 16)

    tf = self.transfer_function
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
  def rgba_to_colormap(self, colormap):

    c = self.colormap_components()
    from numpy import empty
    cmap = empty((colormap.shape[0],len(c)), colormap.dtype)
    for i,ci in enumerate(c):
      cmap[:,i] = colormap[:,ci]
    return cmap

  # ---------------------------------------------------------------------------
  # Tuple of colormap component numbers 0=R, 1=G, 2=B, 3=A for mapping RGBA
  # to a format appropriate for color mode.
  #
  def colormap_components(self):

    m = self.c_mode
    if m.startswith('rgba'):    c = (0,1,2,3)  # RGBA
    elif m.startswith('rgb'):   c = (0,1,2)    # RGB
    elif m.startswith('la'):    c = (0,3)      # RA
    elif m.startswith('l'):     c = (0,)       # R
    else:                       c = (0,1,2,3)  # RGBA
    return c

  # ---------------------------------------------------------------------------
  # 
  def auto_color_mode(self):

    cm = self.color_mode
    auto = cm.startswith('auto')
    opaque = cm.startswith('opaque')
    if auto or opaque:
      from numpy import array
      tf = array(self.transfer_function)
      if len(tf) == 0 or hasattr(self, 'mask_colors'):
        m = 'rgb' if opaque else 'rgba'
      else:
        single_color = colinear(tf[:,2:5], 0.99)
        m = 'l' if single_color else 'rgb'
        if not opaque:
          if self.transparency_thickness != 0:
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
  def luminance_color(self):

    if self.c_mode.startswith('l'):
      ltf, rgba = luminance_transfer_function(self.transfer_function)
    else:
      rgba = (1,1,1,1)
    return rgba

  # ---------------------------------------------------------------------------
  # 
  def auto_projection_mode(self):

    pm = self.projection_mode
    if pm == 'auto':
      sz = self.size
      from chimerax.core.geometry import norm
      spacing = [norm(a) for a in self.transform.axes()]
      s = [n*sp for n,sp in zip(sz, spacing)]
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

    return self.drawing
    
  # ---------------------------------------------------------------------------
  #
  def close_model(self, parent_drawing):

    v = self.drawing
    if not v is None:
      parent_drawing.remove_drawing(v)
    self.drawing = None

# -----------------------------------------------------------------------------
#
def luminance_transfer_function(tf):

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
def colinear(vlist, tolerance = 0.99):

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
def value_type_range(numpy_type):

  from numpy import uint8, int8, uint16, int16
  tsize = {
    uint8: (0, 255),
    int8: (-128, 127),
    uint16: (0, 65535),
    int16: (-32768, 32767),
    }
  return tsize.get(numpy_type, (None, None))
