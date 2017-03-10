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
from ..graphics import Drawing
class GrayScaleDrawing(Drawing):

  def __init__(self, name, blend_manager = None):

    Drawing.__init__(self, name)

    self.grid_size = None
    self.color_grid = None
    self.get_color_plane = None
    self.texture_planes = {}    # maps plane index to texture id
    self.update_colors = False
    self._blend_manager = blend_manager	# ImageBlendManager to blend colors with other drawings,
    if blend_manager:			# is None for BlendedImage.
      blend_manager.add_drawing(self)
    self.planes_drawing = None

    # Mode names rgba4, rgba8, rgba12, rgba16, rgb4, rgb8, rgb12, rgb16,
    #   la4, la8, la12, la16, l4, l8, l12, l16
    self.color_mode = 'rgba8'	# OpenGL texture internal format.

    self.mod_rgba = (1,1,1,1)	# For luminance color modes.

    from ..geometry import Place
    self.ijk_to_xyz = Place(((1,0,0,0),(0,1,0,0),(0,0,1,0)))

    # Use 2d textures along chosen axes or 3d textures.
    # Mode names 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
    self._projection_mode = '2d-z'

    self.maximum_intensity_projection = False

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
  
  def modulation_rgba(self):
    return self.mod_rgba
  def set_modulation_rgba(self, rgba):
    if rgba != self.mod_rgba:
      self.mod_rgba = rgba
      rgba8 = tuple(int(255*r) for r in rgba)
      for p in self.child_drawings():
        p.color = rgba8

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
      
  def set_grid_size(self, grid_size):
    if grid_size != self.grid_size:
      self.remove_planes()
      self.grid_size = grid_size
      bi = self.blend_image
      if bi:
        bi.set_grid_size(grid_size)

  def draw(self, renderer, place, draw_pass, selected_only = False):
    self.update_blend_groups()
    bi = self.blend_image
    if bi:
      if self is bi.master_drawing:
        bi.draw(renderer, place, draw_pass, selected_only)
      return

    from ..graphics import Drawing
    dopaq = (draw_pass == Drawing.OPAQUE_DRAW_PASS and not 'a' in self.color_mode)
    dtransp = (draw_pass == Drawing.TRANSPARENT_DRAW_PASS and 'a' in self.color_mode)
    if not dopaq and not dtransp:
      return

    # Create or update the planes.
    pd = self.planes_drawing
    if pd is None:
      self.planes_drawing = pd = self.make_planes()
    elif self.update_colors:
      self.reload_textures()
    self.update_colors = False

    # Compare stack z axis to view direction to decide whether to reverse plane drawing order.
    r = renderer
    paxis = self.ijk_to_xyz.axes()[self.projection_axis()]
    saxis = self.scene_position.apply_without_translation(paxis) # Scene coords
    caxis = r.current_view_matrix.apply_without_translation(saxis) # Camera coords
    pd.multitexture_reverse_order = (caxis[2] < 0)

    max_proj = dtransp and self.maximum_intensity_projection
    if max_proj:
      r.blend_max(True)
    if dtransp:
      r.write_depth(False)

    Drawing.draw(self, r, place, draw_pass, selected_only)

    if dtransp:
      r.write_depth(True)
    if max_proj:
      r.blend_max(False)

  def remove_planes(self):

    self.texture_planes = {}
    pd = self.planes_drawing
    if pd:
      self.remove_drawing(pd)
      self.planes_drawing = None

  def make_planes(self):

    if self._show_box_faces:
      d = self.make_box_faces()
    elif self._show_ortho_planes:
      d = self.make_ortho_planes()
    else:
      d = self.make_axis_planes(self.projection_axis())
    return d

  def projection_axis(self):
    pmode = self.projection_mode
    axis = {'2d-x': 0, '2d-y': 1, '2d-z': 2}.get(pmode, 2)
    return axis

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

    gs = self.grid_size
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
      self.ijk_to_xyz.move(vap)
      ta[2*p:2*(p+1),:] = tap + 4*p
      textures.append(self.texture_plane(k, axis))
      tc[4*p:4*(p+1),:] = (tc2 if axis == 1 else tc1)

    p = self.new_drawing()
    p.color = tuple(int(255*r) for r in self.modulation_rgba())
    p.use_lighting = False
    p.opaque_texture = (not 'a' in self.color_mode)
    p.geometry = va, ta
    p.texture_coordinates = tc
    p.multitexture = textures
    p.planes = planes

    return p

  def texture_plane(self, k, axis):

    t = self.texture_planes.get((k,axis))
    if t is None:
      d = self.color_plane(k, axis)
      from ..graphics import Texture
      t = Texture(d)
      self.texture_planes[(k,axis)] = t
    return t

  def color_plane(self, k, axis):

    if not self.color_grid is None:
      if axis == 2:
        p = self.color_grid[k,:,:,:]
      elif axis == 1:
        p = self.color_grid[:,k,:,:]
      elif axis == 0:
        p = self.color_grid[:,:,k,:]
    elif self.get_color_plane:
      p = self.get_color_plane(axis, k)
    else:
      p = None

    return p

  def reload_textures(self):

    d = self.planes_drawing
    mtex = d.multitexture
    for t,(k,axis) in enumerate(d.planes):
      data = self.color_plane(k,axis)
      mtex[t].reload_texture(data)

# ---------------------------------------------------------------------------
#
class BlendedImage(GrayScaleDrawing):

  def __init__(self, drawings):

    name = 'blend ' + ', '.join(d.name for d in drawings)
    GrayScaleDrawing.__init__(self, name)

    d = drawings[0]

    self.grid_size = d.grid_size
    self.ijk_to_xyz = d.ijk_to_xyz

    self.drawings = drawings
    for d in drawings:
      d.remove_planes()	# Free textures and opengl buffers

    self._rgba8_array = None

  def draw(self, renderer, place, draw_pass, selected_only = False):
    self.check_update_colors()
    GrayScaleDrawing.draw(self, renderer, place, draw_pass, selected_only)

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
        elif cmode == 'la8':
          copy_la_to_rgba(dp, d.mod_rgba, p)
        else:
          raise ValueError('Cannot blend with color mode %s' % cmode)
      else:
        if cmode == 'rgba8':
          blend_rgba(dp, p)
        elif cmode == 'la8':
          blend_la_to_rgba(dp, d.mod_rgba, p)
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
        # Need to have matching grid size, scene position and grid spacing
        k = (tuple(d.grid_size), tuple(d.scene_position.matrix.flat), tuple(d.ijk_to_xyz.matrix.flat))
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
