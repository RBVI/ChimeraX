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

# ----------------------------------------------------------------------------
# Stack of transparent rectangles.
#
from chimerax.core.graphics import Drawing
class GrayScaleDrawing(Drawing):

  def __init__(self, name, blend_manager = None):

    Drawing.__init__(self, name)

    self.grid_size = None
    self.color_grid = None
    self.get_color_plane = None
    self.update_colors = False
    self._blend_manager = blend_manager	# ImageBlendManager to blend colors with other drawings,
    if blend_manager:			# is None for BlendedImage.
      blend_manager.add_drawing(self)
    self._multiaxis_planes = [None, None, None]	# For x, y, z axis projection
    self._planes_drawing = None			# For ortho and box mode display
    self._view_aligned_planes = None		# ViewAlignedPlanes instance for 3d projection mode

    # Mode names rgba4, rgba8, rgba12, rgba16, rgb4, rgb8, rgb12, rgb16,
    #   la4, la8, la12, la16, l4, l8, l12, l16
    self.color_mode = 'rgba8'	# OpenGL texture internal format.

    self.mod_rgba = (1,1,1,1)	# For luminance color modes.

    from chimerax.core.geometry import Place
    self.ijk_to_xyz = Place(((1,0,0,0),(0,1,0,0),(0,0,1,0)))

    # Use 2d textures along chosen axes or 3d textures.
    # Mode names 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
    self._projection_mode = '2d-z'

    self._maximum_intensity_projection = False

    self.linear_interpolation = True

    self.SRC_1_DST_1_MINUS_ALPHA = '1,1-a'
    self.SRC_ALPHA_DST_1_MINUS_ALPHA = 'a,1-a'
    self.transparency_blend_mode = self.SRC_ALPHA_DST_1_MINUS_ALPHA

    self.brightness_and_transparency_correction = True

    self.minimal_texture_memory = False

    self.show_outline_box = False
    self.outline_box_rgb = (1,1,1,1)
    self.outline_box_linewidth = 1

    self._show_box_faces = False

    self._show_ortho_planes = 0  # bits 0,1,2 correspond to axes x,y,z.
    self._ortho_planes_position = (0,0,0)

  def delete(self):
    b = self._blend_manager
    if b:
      b.remove_drawing(self)
    self.remove_planes()
    Drawing.delete(self)

  @property
  def blend_image(self):
    b = self._blend_manager
    return b.blend_image(self) if b else None

  def update_blend_groups(self):
    b = self._blend_manager
    if b:
      b.update_groups()

  def _get_projection_mode(self):
    return self._projection_mode
  def _set_projection_mode(self, mode):
    if mode != self._projection_mode:
      self._projection_mode = mode
      self.remove_planes()
  projection_mode = property(_get_projection_mode, _set_projection_mode)
    
  def shown_orthoplanes(self):
    return self._show_ortho_planes
  def set_shown_orthoplanes(self, s):
    if s != self._show_ortho_planes:
      self._show_ortho_planes = s
      self.remove_planes()
  show_ortho_planes = property(shown_orthoplanes, set_shown_orthoplanes)

  def orthoplanes_position(self):
    return self._ortho_planes_position
  def set_orthoplanes_position(self, p):
    if tuple(p) != self._ortho_planes_position:
      self._ortho_planes_position = tuple(p)
      self.remove_planes()	# TODO: Reuse current planes moving to new position.
  ortho_planes_position = property(orthoplanes_position, set_orthoplanes_position)

  def showing_box_faces(self):
    return self._show_box_faces
  def set_showing_box_faces(self, s):
    if s != self._show_box_faces:
      self._show_box_faces = s
      self.remove_planes()
  show_box_faces = property(showing_box_faces, set_showing_box_faces)

  @property
  def _showing_view_aligned(self):
    return (self.projection_mode == '3d'
            and not self._show_ortho_planes
            and not self._show_box_faces)
  
  def max_intensity_projection(self):
    return self._maximum_intensity_projection
  def set_max_intensity_projection(self, enable):
    if enable != self._maximum_intensity_projection:
      self._maximum_intensity_projection = enable
  maximum_intensity_projection = property(max_intensity_projection, set_max_intensity_projection)
  
  def modulation_rgba(self):
    return self.mod_rgba
  def set_modulation_rgba(self, rgba):
    if rgba != self.mod_rgba:
      self.mod_rgba = rgba
      rgba8 = tuple(int(255*r) for r in rgba)
      for p in self.child_drawings():
        p.color = rgba8

  @property
  def modulation_color(self):
    return tuple(int(255*r) for r in self.modulation_rgba())
  
  # 3 by 4 matrix mapping grid indices to xyz coordinate space.
  def array_coordinates(self):
    return self.ijk_to_xyz
  def set_array_coordinates(self, tf):
    if tf != self.ijk_to_xyz:
      self.ijk_to_xyz = tf
      # TODO: Just update vertex buffer.
      self.remove_planes()
      bi = self.blend_image
      if bi:
        bi.set_array_coordinates(tf)

  def set_volume_colors(self, color_values):	# uint8 or float
    self.color_grid = color_values
    grid_size = tuple(color_values.shape[2::-1])
    self.set_grid_size(grid_size)
    self.update_colors = True

  # Callback takes axis and plane numbers and returns color plane.
  # This is an alternative to setting a 3-d color array that reduceso
  # memory use.
  def set_color_plane_callback(self, grid_size, get_color_plane):
    self.get_color_plane = get_color_plane
    self.set_grid_size(grid_size)
    self.update_colors = True

  def set_color_mode(self, color_mode):
    if color_mode != self.color_mode:
      self.color_mode = color_mode
      self.remove_planes()

  @property
  def opaque(self):
    return (not 'a' in self.color_mode)
  
  def set_linear_interpolation(self, lin_interp):
    '''Can only call this when OpenGL context current.'''
    if lin_interp != self.linear_interpolation:
      self.linear_interpolation = lin_interp
      self.remove_planes()
        
  def set_grid_size(self, grid_size):
    if grid_size != self.grid_size:
      self.remove_planes()
      self.grid_size = grid_size
      bi = self.blend_image
      if bi:
        bi.set_grid_size(grid_size)

  def bounds(self):
    # Override bounds because GrayScaleDrawing does not set geometry until draw() is called
    # but setting initial camera view uses bounds before draw() is called.

    if not self.display:
      return None

    gs = self.grid_size
    if gs is None:
      return None

    gx, gy, gz = gs
    from numpy import array, float32
    corners = array(((0,0,0), (gx,0,0), (0,gy,0), (0,0,gz), (gx,gy,0), (gx,0,gz), (0,gy,gz), (gx,gy,gz)), float32)
    corners[:] += -.5
    from chimerax.core.geometry import point_bounds
    b = point_bounds(corners, self.get_scene_positions(displayed_only = True) * self.ijk_to_xyz)
    return b
    
  def drawings_for_each_pass(self, pass_drawings):
    '''Override Drawing method because geometry is not set until draw() is called.'''
    if not self.display:
      return
        
    transparent = ('a' in self.color_mode)
    p = self.TRANSPARENT_DRAW_PASS if transparent else self.OPAQUE_DRAW_PASS
    if p in pass_drawings:
      pass_drawings[p].append(self)
    else:
      pass_drawings[p] = [self]

    # Do not include child drawings since this drawing overrides draw() method
    # and draws the children.

  def draw(self, renderer, draw_pass):
    if not self.display:
      return

    self.update_blend_groups()
    bi = self.blend_image
    if bi:
      if self is bi.master_drawing:
        bi.draw(renderer, draw_pass)
      return

    transparent = ('a' in self.color_mode)
    from chimerax.core.graphics import Drawing
    dopaq = (draw_pass == Drawing.OPAQUE_DRAW_PASS and not transparent)
    dtransp = (draw_pass == Drawing.TRANSPARENT_DRAW_PASS and transparent)
    if not dopaq and not dtransp:
      return

    pd = self._update_planes(renderer)

    if self.update_colors:
      self.reload_textures()
    self.update_colors = False

    self._draw_planes(renderer, draw_pass, dtransp, pd)

  def _draw_planes(self, renderer, draw_pass, dtransp, drawing):
    r = renderer
    max_proj = dtransp and self.maximum_intensity_projection
    if max_proj:
      r.blend_max(True)
    if dtransp:
      r.write_depth(False)
    blend1 = (dtransp and self.transparency_blend_mode == self.SRC_1_DST_1_MINUS_ALPHA)
    if blend1:
      r.blend_alpha(False)

    drawing.draw(r, draw_pass)

    if blend1:
      r.blend_alpha(True)
    if dtransp:
      r.write_depth(True)
    if max_proj:
      r.blend_max(False)

  def _update_planes(self, renderer):
    # Create or update the planes.
    view_dir = self.view_direction(renderer)
    if self._showing_view_aligned:
      self._remove_axis_planes()
      pd = self._update_view_aligned_planes(view_dir)
    else:
      self._remove_view_planes()
      pd = self._update_axis_aligned_planes(view_dir)
    return pd

  def _update_axis_aligned_planes(self, view_direction):
    # Render grid aligned planes
    axis, rev = self.projection_axis(view_direction)
    pd = self._planes_drawing if axis is None else self._multiaxis_planes[axis]
    if pd is None:
      sc = self.shape_changed
      pd = self.make_planes(axis)
      self.update_colors = False
      if axis is None:
        self._planes_drawing = pd
      else:
        if tuple(self._multiaxis_planes) != (None, None, None):
          # Reset shape change flag since this is the same shape.
          self.shape_changed = sc
        self._multiaxis_planes[axis] = pd

    if axis is not None:
      # Reverse drawing order if needed to draw back to front
      pd.multitexture_reverse_order = rev
      sc = self.shape_changed
      for d in self._multiaxis_planes:
        disp = (d is pd)
        if d and d.display != disp:
          # TODO: Make drawing not cause redraw if display value does not change.
          d.display = disp
      # When switching planes, do not set shape change flag
      # since that causes center of rotation update with
      # front center rotation method, which messes up spin movies.
      self.shape_changed = sc

    return pd

  def _update_view_aligned_planes(self, view_direction):
    pd = self._view_aligned_planes
    if pd is None:
      pd = ViewAlignedPlanes(self.color_plane, self.grid_size, self.ijk_to_xyz,
                             self.modulation_color, self.opaque, self.linear_interpolation)
      self.add_drawing(pd)
      pd.load_texture()
      self._view_aligned_planes = pd
      self.update_colors = False
    pd.update_geometry(view_direction, self.scene_position)
    return pd

  def remove_planes(self):
    self._remove_axis_planes()
    self._remove_view_planes()

  def _remove_axis_planes(self):
    pd = self._planes_drawing
    if pd:
      self.remove_drawing(pd)
      self._planes_drawing = None

    for pd in self._multiaxis_planes:
      if pd:
        self.remove_drawing(pd)
    self._multiaxis_planes = [None,None,None]

  def _remove_view_planes(self):
    pd = self._view_aligned_planes
    if pd:
      self.remove_drawing(pd)
      self._view_aligned_planes = None

  def make_planes(self, axis):
    if axis is not None:
      d = self.make_axis_planes(axis)
    elif self._show_box_faces:
      d = self.make_box_faces()
    elif self._show_ortho_planes:
      d = self.make_ortho_planes()
    return d

  def view_direction(self, render):
    return -render.current_view_matrix.inverse().z_axis()	# View direction, scene coords

  def projection_axis(self, view_direction):
    # View matrix maps scene to camera coordinates.
    v = view_direction
    if self._show_box_faces or self._show_ortho_planes or self.projection_mode == '3d':
      return None, False

    # Determine which axis has box planes with largest projected area.
    bx,by,bz = (self.scene_position * self.ijk_to_xyz).axes()	# Box axes, scene coordinates
    # Scale axes to length of box so that plane axis chosen maximizes plane view area for box.
    gs = self.grid_size
    bx *= gs[0]
    by *= gs[1]
    bz *= gs[2]
    from chimerax.core.geometry import cross_product, inner_product
    box_face_normals = [cross_product(by,bz), cross_product(bz,bx), cross_product(bx,by)]
    pmode = self.projection_mode
    if pmode == '2d-xyz':
      view_areas = [inner_product(v,bfn) for bfn in box_face_normals]
      from numpy import argmax, abs
      axis = argmax(abs(view_areas))
      rev = (view_areas[axis] > 0)
    else:
      axis = {'2d-x': 0, '2d-y': 1, '2d-z': 2}.get(pmode, 2)
      rev = (inner_product(v,box_face_normals[axis]) > 0)

    return axis, rev

  def make_axis_planes(self, axis = 2):
    planes = tuple((k, axis) for k in range(0,self.grid_size[axis]))
    d = self.make_planes_drawing(planes)
    return d

  def make_ortho_planes(self):
    op = self._show_ortho_planes
    p = self.ortho_planes_position
    show_axis = (op & 0x1, op & 0x2, op & 0x4)
    planes = tuple((p[axis], axis) for axis in (0,1,2) if show_axis[axis])
    d = self.make_planes_drawing(planes)
    return d

  def make_box_faces(self):
    gs = self.grid_size
    planes = (tuple((0,axis) for axis in (0,1,2)) +
              tuple((gs[axis]-1,axis) for axis in (0,1,2)))
    d = self.make_planes_drawing(planes)
    return d

  # Each plane is an index position and axis (k,axis).
  def make_planes_drawing(self, planes):
    pd = AxisAlignedPlanes(planes, self.grid_size, self.ijk_to_xyz, self.color_plane,
                           self.modulation_color, self.opaque, self.linear_interpolation)
    self.add_drawing(pd)
    return pd

  def reload_textures(self):
    pd = self._view_aligned_planes
    if pd:
      pd.load_texture()

    for pd in self._multiaxis_planes + [self._planes_drawing]:
      if pd:
        pd.load_textures()

  def color_plane(self, k, axis, view_aligned=False):
    if self.color_grid is not None:
      if axis == 2:
        p = self.color_grid[k,:,:,:]
      elif axis == 1:
        p = self.color_grid[:,k,:,:]
      elif axis == 0:
        p = self.color_grid[:,:,k,:]
    elif self.get_color_plane:
      p = self.get_color_plane(k, axis, view_aligned=view_aligned)
    else:
      p = None

    return p

# ---------------------------------------------------------------------------
#
class AxisAlignedPlanes(Drawing):
  
  def __init__(self, planes, grid_size, ijk_to_xyz, color_plane,
               modulation_color, opaque, linear_interpolation):
    name = 'grayscale axis aligned planes'
    Drawing.__init__(self, name)

    self.color = modulation_color
    self.use_lighting = False
    self.opaque_texture = opaque
    self.linear_interpolation = linear_interpolation

    self._color_plane = color_plane
    self._set_planes(planes, grid_size, ijk_to_xyz, color_plane)
    
  def _set_planes(self, planes, grid_size, ijk_to_xyz, color_plane):
    gs = grid_size
    from numpy import array, float32, int32, empty
    tap = array(((0,1,2),(0,2,3)), int32)
    tc1 = array(((0,0),(1,0),(1,1),(0,1)), float32)
    tc2 = array(((0,0),(0,1),(1,1),(1,0)), float32)
    np = len(planes)
    va = empty((4*np,3), float32)
    tc = empty((4*np,2), float32)
    ta = empty((2*np,3), int32)
    textures = []
    for p, (k,axis) in enumerate(planes):
      vap = va[4*p:,:]
      vap[:,:] = -0.5
      vap[:,axis] = k
      a0, a1 = (axis + 1) % 3, (axis + 2) % 3
      vap[1:3,a0] += gs[a0]
      vap[2:4,a1] += gs[a1]
      ijk_to_xyz.transform_points(vap, in_place = True)
      ta[2*p:2*(p+1),:] = tap + 4*p
      d = color_plane(k, axis)
      textures.append(self._plane_texture(d))
      tc[4*p:4*(p+1),:] = (tc2 if axis == 1 else tc1)

    self.set_geometry(va, None, ta)
    self.texture_coordinates = tc
    self.multitexture = textures
    self.planes = planes

  def _plane_texture(self, colors):
    dc = colors.copy()	# Data array may be reused before texture is filled so copy it.
    from chimerax.core.graphics import Texture
    t = Texture(dc)
    t.linear_interpolation = self.linear_interpolation
    return t

  def load_textures(self):
    mtex = self.multitexture
    for t,(k,axis) in enumerate(self.planes):
      data = self._color_plane(k, axis)
      mtex[t].reload_texture(data, now = True)

# ---------------------------------------------------------------------------
#
class ViewAlignedPlanes(Drawing):
  
  def __init__(self, color_plane, grid_size, ijk_to_xyz,
               modulation_color, opaque, linear_interpolation):

    name = 'grayscale view aligned planes'
    Drawing.__init__(self, name)

    self._color_plane = color_plane
    self._grid_size = grid_size
    self._ijk_to_xyz = ijk_to_xyz
    self._compute_geometry_constants()
    self.color = modulation_color
    self.use_lighting = False
    self.opaque_texture = opaque
    self._linear_interpolation = linear_interpolation
    self._last_view_direction = None

  def update_geometry(self, view_direction, scene_position):
    tvd = tuple(view_direction)
    if tvd == self._last_view_direction:
      return
    self._last_view_direction = tvd
    
    va, tc, ta = self._perp_planes_geometry(view_direction, scene_position)
    self.set_geometry(va, None, ta)
    self.texture_coordinates = tc

  def _compute_geometry_constants(self):
    ei,ej,ek = [i-1 for i in self._grid_size]
    grid_corners = ((0,0,0),(ei,0,0),(0,ej,0),(ei,ej,0),(0,0,ek),(ei,0,ek),(0,ej,ek),(ei,ej,ek))
    self._corners = self._ijk_to_xyz * grid_corners	# in volume coords
    # Use view aligned spacing equal to minimum grid spacing along 3 axes.
    self._plane_spacing = min(self._ijk_to_xyz.axes_lengths())
    # Use texture coord range [1/2n,1-1/2n], not [0,1].
    from chimerax.core.geometry.place import scale, translation
    v_to_tc = scale((1/(ei+1), 1/(ej+1), 1/(ek+1))) * translation((0.5,0.5,0.5)) * self._ijk_to_xyz.inverse()
    self._vertex_to_texcoord = v_to_tc
    
  def _perp_planes_geometry(self, view_direction, scene_position):

    if scene_position.is_identity():
      axis = -view_direction
    else:
      axis = -scene_position.transpose().transform_vector(view_direction)

    # Find number of cut planes
    corners = self._corners
    from . import offset_range
    omin, omax = offset_range(corners, axis)
    spacing = self._plane_spacing
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

  def load_texture(self):
    t = self.texture
    if t is None:
      self.texture = t = self._texture_3d()
    td = self._texture_3d_data()
    t.reload_texture(td, now = True)

  def _texture_3d(self):
    td = self._texture_3d_data()
    from chimerax.core.graphics import Texture
    t = Texture(td, dimension = 3)
    t.linear_interpolation = self._linear_interpolation
    return t

  def _texture_3d_data(self):
    z_axis = 2
    p = self._color_plane(0, z_axis, view_aligned=True)
    sz = self._grid_size[z_axis]
    if sz == 1:
      td = p
    else:
      from numpy import empty
      td = empty((sz,) + tuple(p.shape), p.dtype)
      td[0,:] = p
      for k in range(1,sz):
        td[k,:] = self._color_plane(k, z_axis, view_aligned=True)
    return td

# ---------------------------------------------------------------------------
#
class BlendedImage(GrayScaleDrawing):

  def __init__(self, drawings):

    name = 'blend ' + ', '.join(d.name for d in drawings)
    GrayScaleDrawing.__init__(self, name)

    self.drawings = drawings

    self.mirror_attributes()

    for d in drawings:
      d.remove_planes()	# Free textures and opengl buffers

    self._rgba8_array = None

  def mirror_attributes(self):
    d = self.drawings[0]
    if 'a' not in d.color_mode:
      self.color_mode = 'rgb8'

    for attr in ('grid_size', 'ijk_to_xyz', 'projection_mode', 'maximum_intensity_projection',
                 'linear_interpolation', 'transparency_blend_mode', 'brightness_and_transparency_correction',
                 'minimal_texture_memory', 'show_outline_box', 'outline_box_rgb', 'outline_box_linewidth',
                 'show_box_faces', 'show_ortho_planes', 'ortho_planes_position'):
      setattr(self, attr, getattr(d, attr))

  def draw(self, renderer, draw_pass):

    self.mirror_attributes()
    self.check_update_colors()
    
    GrayScaleDrawing.draw(self, renderer, draw_pass)

  @property
  def master_drawing(self):
      return self.drawings[0]

  def color_plane(self, k, axis):
    p = None
    for d in self.drawings:
      dp = d.color_plane(k, axis)
      d.update_colors = False
      cmode = d.color_mode
      if p is None:
        h,w = dp.shape[:2]
        p = self.rgba8_array(w,h)
        if cmode == 'rgba8':
          p[:] = dp
        elif cmode == 'rgb8':
          p[:,:,:3] = dp
          p[:,:,3] = 255
        elif cmode == 'la8':
          copy_la_to_rgba(dp, d.mod_rgba, p)
        elif cmode == 'l8':
          copy_l_to_rgba(dp, d.mod_rgba, p)
        else:
          raise ValueError('Cannot blend with color mode %s' % cmode)
      else:
        if cmode == 'rgba8':
          blend_rgba(dp, p)
        if cmode == 'rgb8':
          blend_rgb_to_rgba(dp, p)
        elif cmode == 'la8':
          blend_la_to_rgba(dp, d.mod_rgba, p)
        elif cmode == 'l8':
          blend_l_to_rgba(dp, d.mod_rgba, p)
    return p

  def rgba8_array(self, w, h):
    # Reuse same array for faster color updating.
    a = self._rgba8_array
    if a is None or tuple(a.shape) != (h, w, 4):
      from numpy import empty, uint8
      self._rgba8_array = a = empty((h,w,4), uint8)
    return a

  def check_update_colors(self):
    for d in self.drawings:
      if d.update_colors:
        self.update_colors = True
        d.update_colors = False

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
    self.drawing_blend_image = {}	# Map drawing to BlendedImage
    self.need_group_update = False

  def add_drawing(self, d):
    self.drawing_blend_image[d] = None
    self.need_group_update = True

  def remove_drawing(self, d):
    dbi = self.drawing_blend_image
    bi = dbi.get(d)
    if bi:
      for d2 in bi.drawings:
        dbi[d2] = None
      self.blend_images.discard(bi)
      bi.delete()
    del dbi[d]
    self.need_group_update = True

  def blend_image(self, drawing):
    return self.drawing_blend_image.get(drawing, None)

  def update_groups(self):
    if not self.need_group_update:
      return
    self.need_group_update = False

    # TODO: Don't update groups unless drawing changed.
    groups = []
    dbi = self.drawing_blend_image
    drawings = list(dbi.keys())
    drawings.sort(key = lambda d: d.name)
    aligned = {}
    for d in drawings:
      if d.display and d.parents_displayed and not d.maximum_intensity_projection:
        sortho = d._show_ortho_planes
        orthoplanes = (sortho, tuple(d._ortho_planes_position)) if sortho else sortho
        # Need to have matching grid size, scene position, grid spacing, box face mode, orthoplane mode
        k = (tuple(d.grid_size), tuple(d.scene_position.matrix.flat), tuple(d.ijk_to_xyz.matrix.flat),
             d._show_box_faces, orthoplanes)
        if k in aligned:
          aligned[k].append(d)
        else:
          g = [d]
          groups.append(g)
          aligned[k] = g

    dgroup = {}
    for g in groups:
      if len(g) >= 2:
        for d in g:
          dgroup[d] = g

    # Remove blend images for groups that no longer exist.
    bi_gone = []
    bis = self.blend_images
    for bi in bis:
      d = bi.drawings[0]
      if dgroup.get(d,None) != bi.drawings:
        bi_gone.append(bi)
    for bi in bi_gone:
      for d in bi.drawings:
        dbi[d] = None
      bis.discard(bi)
      bi.delete()
      
    # Created blend images for new groups
    for g in groups:
      if len(g) >= 2:
        if dbi[g[0]] is None:
          bi = BlendedImage(g)
          bis.add(bi)
          for d in g:
            dbi[d] = bi

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
