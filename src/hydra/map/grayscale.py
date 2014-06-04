# ----------------------------------------------------------------------------
#
class Gray_Scale_Drawing:

  def __init__(self, surface):

    self.grid_size = None
    self.color_grid = None
    self.get_color_plane = None
    self.surface = surface      # Stack of rectangles
    self.plane_drawings = None
    self.texture_planes = {}    # maps plane index to texture id
    self.update_colors = False

    # Mode names rgba4, rgba8, rgba12, rgba16, rgb4, rgb8, rgb12, rgb16,
    #   la4, la8, la12, la16, l4, l8, l12, l16
    self.color_mode = 'rgba8'	# OpenGL texture internal format.

    self.mod_rgba = (1,1,1,1)	# For luminance color modes.

    from ..geometry.place import Place
    self.ijk_to_xyz = Place(((1,0,0,0),(0,1,0,0),(0,0,1,0)))

    # Use 2d textures along chosen axes or 3d textures.
    # Mode names 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
    self.projection_mode = '2d-z'

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

    self.show_box_faces = False

    self.show_ortho_planes = 0  # bits 0,1,2 correspond to axes x,y,z.
    self.orthoPlanesPosition = (0,0,0)

  def shown_orthoplanes(self):
    return self.show_ortho_planes
  def set_shown_orthoplanes(self, s):
    if s != self.show_ortho_planes:
      self.show_ortho_planes = s
      self.delete()
  showOrthoPlanes = property(shown_orthoplanes, set_shown_orthoplanes)

  def showing_box_faces(self):
    return self.show_box_faces
  def set_showing_box_faces(self, s):
    if s != self.show_box_faces:
      self.show_box_faces = s
      self.delete()
  showBoxFaces = property(showing_box_faces, set_showing_box_faces)
  
  def modulation_rgba(self):
    return self.mod_rgba
  def set_modulation_rgba(self, rgba):
    self.mod_rgba = rgba
      
  # 3 by 4 matrix mapping grid indices to xyz coordinate space.
  def array_coordinates(self):
    return self.ijk_to_xyz
  def set_array_coordinates(self, tf):
    self.ijk_to_xyz = tf
    # TODO: Just update vertex buffer.
    self.delete()

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

  def set_grid_size(self, grid_size):
    if grid_size == self.grid_size:
      return

    self.delete()
    self.grid_size = grid_size

  def texture_id(self):	  # Get 3d texture id.
    return 0

  def texture_matrix(self):	# Maps local coordinates to texture coords.
    pass

  def draw(self, renderer, camera_view, draw_pass):

    from ..graphics import Drawing
    dopaq = (draw_pass == Drawing.OPAQUE_DRAW_PASS and not 'a' in self.color_mode)
    dtransp = (draw_pass == Drawing.TRANSPARENT_DRAW_PASS and 'a' in self.color_mode)
    if not dopaq and not dtransp:
      return

    if self.plane_drawings is None:
      self.plane_drawings = self.make_planes()
    elif self.update_colors:
      self.reload_textures()
      self.update_colors = False

    zaxis = self.ijk_to_xyz.z_axis()
    czaxis = camera_view.apply_without_translation(zaxis) # z axis in camera coords
    reverse = (czaxis[2] < 0)

    spieces = self.plane_drawings
    plist = spieces[::-1] if reverse else spieces

    Drawing.draw(self.surface, renderer, camera_view, draw_pass, children = plist)

  def show(self):

    if self.plane_drawings:
      for p in self.plane_drawings:
        p.display = True

  def hide(self):

    if self.plane_drawings:
      for p in self.plane_drawings:
        p.display = False

  def delete(self):
    plist = self.plane_drawings
    if plist is None:
      return

    self.texture_planes = {}
    self.surface.remove_drawings(plist)
    self.plane_drawings = None

  def make_planes(self):

    if self.show_box_faces:
      plist = self.make_box_faces()
    elif self.show_ortho_planes:
      plist = self.make_ortho_planes()
    else:
      plist = self.make_axis_planes()
    return plist

  def make_axis_planes(self, axis = 2):

    planes = tuple((k, axis) for k in range(0,self.grid_size[axis]))
    plist = self.make_plane_drawings(planes)
    return plist

  def make_ortho_planes(self):
    
    op = self.show_ortho_planes
    p = self.orthoPlanesPosition
    show_axis = (op & 0x1, op & 0x2, op & 0x4)
    planes = tuple((p[axis], axis) for axis in (0,1,2) if show_axis[axis])
    plist = self.make_plane_drawings(planes)
    return plist

  def make_box_faces(self):
    
    gs = self.grid_size
    planes = (tuple((0,axis) for axis in (0,1,2)) +
              tuple((gs[axis]-1,axis) for axis in (0,1,2)))
    plist = self.make_plane_drawings(planes)
    return plist

  # Each plane is an index position and axis (k,axis).
  def make_plane_drawings(self, planes):

    s = self.surface
    gs = self.grid_size
    from numpy import array, float32, int32, empty
    ta = array(((0,1,2),(0,2,3)), int32)
    tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
    tc1 = array(((0,0),(0,1),(1,1),(1,0)), float32)
    plist = []
    for k, axis in planes:
      va = empty((4,3), float32)
      va[:,:] = -0.5
      va[:,axis] = k
      a0, a1 = (axis + 1) % 3, (axis + 2) % 3
      va[1:3,a0] += gs[a0]
      va[2:4,a1] += gs[a1]
      self.ijk_to_xyz.move(va)
      p = s.new_drawing()
      p.geometry = va, ta
      p.color = self.modulation_rgba()
      p.use_lighting = False
      p.texture = self.texture_plane(k, axis)
      p.texture_coordinates = tc1 if axis == 1 else tc
      p.opaque_texture = (not 'a' in self.color_mode)
      p.plane = (k,axis)
      plist.append(p)

    return plist

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

    if self.plane_drawings is None:
      return

    for p in self.plane_drawings:
      t = p.texture
      k,axis = p.plane
      data = self.color_plane(k,axis)
      t.reload_texture(data)
