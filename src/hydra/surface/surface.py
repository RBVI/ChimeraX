class Surface:
  '''
  A Surface represents one or more sets of triangles in 3 dimensional space, each set being a Surface_Piece.
  Surfaces are used to draw molecules, density maps, geometric shapes and other models.
  A Surface has a name, a unique id number which is a positive integer, it can be displayed or hidden,
  has a placement in space, or multiple copies can be placed in space, and a surface can be selected.
  The coordinates, colors, normal vectors and other geometric and display properties are managed by the
  Surface_Piece objects.  Individual surface pieces can be added or removed.  The purpose of pieces is
  for convenience in adding and removing parts of a surface.
  '''

  def __init__(self, name):
    self.name = name
    self.id = None              # positive integer
    self._display = True       # private. use display property
    from ..geometry.place import Place
    self.placement = Place()
    self.copies = []
    self.plist = []
    self.selected = False
    self.redraw_needed = False
    self.__destroyed__ = False

  def surface_pieces(self):
    '''Return the list of surface pieces.'''
    return self.plist

  def new_piece(self, name = None):
    '''Create a new empty surface piece.'''
    p = Surface_Piece(name)
    p.surface = self
    self.plist.append(p)
    self.redraw_needed = True
    return p

  def remove_piece(self, p):
    '''Delete a specified surface piece.'''
    self.plist.remove(p)
    p.delete()
    self.redraw_needed = True

  def remove_pieces(self, pieces):
    '''Delete specified surface pieces.'''
    pset = set(pieces)
    self.plist = [p for p in self.plist if not p in pset]
    for p in pieces:
      p.delete()
    self.redraw_needed = True

  def remove_all_pieces(self):
    '''Delete all surface pieces.'''
    self.remove_pieces(self.plist)

  def get_display(self):
    return self._display
  def set_display(self, display):
    self._display = display
    self.redraw_needed = True
  display = property(get_display, set_display)
  '''Whether or not the surface is drawn.'''

  def get_place(self):
    return self.placement
  def set_place(self, place):
    self.placement = place
    self.redraw_needed = True
  place = property(get_place, set_place)
  '''Position and orientation of the surface in space.'''

  def showing_transparent(self):
    '''Are any transparent surface pieces being displayed.'''
    for p in self.plist:
      if p.display and not p.opaque():
        return True
    return False

  def draw(self, viewer, draw_pass, reverse_order = False):
    '''Draw all displayed surface pieces in the specified view using the given draw pass.'''
    plist = self.plist[::-1] if reverse_order else self.plist
    self.draw_pieces(plist, viewer, draw_pass)

  def draw_pieces(self, plist, viewer, draw_pass):
    '''Draw the specified surface pieces in a view using the given draw pass.'''
    if draw_pass == viewer.OPAQUE_DRAW_PASS:
      for p in plist:
        if p.display and p.opaque():
          p.draw(viewer)
    elif draw_pass in (viewer.TRANSPARENT_DRAW_PASS, viewer.TRANSPARENT_DEPTH_DRAW_PASS):
      ptransp = [p for p in plist if p.display and not p.opaque()]
      if ptransp:
        for p in ptransp:
          p.draw(viewer)

  def bounds(self):
    '''
    The bounds of all surface pieces including undisplayed ones in the
    coordinate system of the surface.  Does not include copies.
    '''
    from ..geometry.bounds import union_bounds
    return union_bounds(p.bounds() for p in self.plist)

  def placed_bounds(self):
    '''
    The bounds of all surface pieces including undisplayed ones in the
    global coordinate system and including surface copies.
    '''
    b = self.bounds()
    if b is None or b == (None, None):
      return None
    if self.copies:
      copies = self.copies
    elif not self.placement.is_identity(tolerance = 0):
      copies = [self.placement]
    else:
      return b
    from ..geometry import bounds
    return bounds.copies_bounding_box(b, copies)

  def first_intercept(self, mxyz1, mxyz2):
    '''
    Find the first intercept of a line segment with the displayed pieces of the surface.
    Return the fraction of the distance along the segment where the intersection occurs
    and a Surface_Piece_Selection object for the intercepted piece.  For no intersection
    two None values are returned.  This routine is used to determine the front-most point
    in the center of view to be used as the interactive center of rotation.
    '''
    f = None
    sp = None
    # TODO handle surface model copies.
    from .. import _image3d
    for p in self.plist:
      if p.display:
        fmin = p.first_intercept(mxyz1, mxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
          sp = p
    return f, Surface_Piece_Selection(sp)

  def delete(self):
    '''
    Delete all surface pieces.
    '''
    self.remove_all_pieces()
  
class Surface_Piece(object):
  '''
  Surface_Pieces are created by the Surface new_piece() method and represent a set of triangles
  that can be added or removed from a Surface independent of other sets of triangles.  The basic
  data defining the triangles is an N by 3 array of vertices (float32 numpy array) and an array
  that defines triangles by specify 3 integer index values into the vertex array which define the
  3 corners of a triangle.  The triangle array is of shape T by 3 and is a numpy int32 array.
  The filled triangles or a mesh consisting of just the triangle edges can be shown.
  The vertices can be individually colored with linear interpolation of colors across triangles,
  or all triangles can be given the same color, or a 2-dimensional texture can be used to color
  the triangles with texture coordinates assigned to the vertices.  Transparency values can be assigned
  to the vertices. Individual triangles or triangle edges in mesh display style can be hidden.
  An N by 3 float array gives   normal vectors, one normal per vertex, for lighting calculations 
  when drawing the surface.  Multiple copies of the surface piece can be drawn with each specified
  by a position and orientation.  Copies can alternatively be specified by a shift and scale factor
  but no rotation, useful for copies of spheres.  Each copy can have its own single color, or all
  copies can use the same per-vertex or texture coloring.  Rendering of surface pieces is done with
  OpenGL using the draw module.
  '''

  Solid = 'solid'
  Mesh = 'mesh'
  Dot = 'dot'

  def __init__(self, name = None):
    self.name = name
    self.vertices = None
    self.triangles = None
    self.normals = None
    self.shift_and_scale = None         # Instance copies
    self.copies34 = []                  # Instance matrices, 3x4
    self.copies44 = None                # Instance matrices, 4x4 opengl
    self.vertex_colors = None
    self.instance_colors = None         # N by 4 uint8 values
    self.edge_mask = None
    self.masked_edges = None
    self.display = True
    self.display_style = self.Solid
    self.color_rgba = (.7,.7,.7,1)
    self.texture = None
    self.texture_coordinates = None
    self.opaque_texture = False
    self.__destroyed__ = False

    self.vao = None     	# Holds the buffer pointers and bindings

    # Surface piece attribute name, shader variable name, instancing
    from .. import draw
    from numpy import uint32, uint8
    bufs = (('vertices', draw.VERTEX_BUFFER),
            ('normals', draw.NORMAL_BUFFER),
            ('shift_and_scale', draw.INSTANCE_SHIFT_AND_SCALE_BUFFER),
            ('copies44', draw.INSTANCE_MATRIX_BUFFER),
            ('vertex_colors', draw.VERTEX_COLOR_BUFFER),
            ('instance_colors', draw.INSTANCE_COLOR_BUFFER),
            ('texture_coordinates', draw.TEXTURE_COORDS_2D_BUFFER),
            ('elements', draw.ELEMENT_BUFFER),
            )
    obufs = []
    for a,v in bufs:
      b = draw.Buffer(v)
      b.surface_piece_attribute_name = a
      obufs.append(b)
    self.opengl_buffers = obufs
    self.element_buffer = obufs[-1]

  def delete(self):
    '''Release all the arrays and graphics memory associated with the surface piece.'''
    self.vertices = None
    self.triangles = None
    self.normals = None
    self.edge_mask = None
    self.texture = None
    self.texture_coordinates = None
    self.masked_edges = None
    for b in self.opengl_buffers:
      b.delete_buffer()

    self.vao = None

  def get_geometry(self):
    return self.vertices, self.triangles
  def set_geometry(self, g):
    self.vertices, self.triangles = g
    self.masked_edges = None
    self.edge_mask = None
    self.surface.redraw_needed = True
  geometry = property(get_geometry, set_geometry)
  '''Geometry is the array of vertices and array of triangles.'''

  def get_copies(self):
    return self.copies34
  def set_copies(self, copies):
    self.copies34 = copies
    self.copies44 = opengl_matrices(copies) if copies else None
    self.surface.redraw_needed = True
  copies = property(get_copies, set_copies)
  '''
  Copies of the surface piece are placed using a 3 by 4 matrix with the first 3 columns
  giving a linear transformation, and the last column specifying a shift.
  '''

  def shader_changed(self, shader):
    return self.vao is None or shader != self.vao.shader

  def bind_buffers(self, shader = None):
    if self.shader_changed(shader):
      from .. import draw
      self.vao = draw.Bindings(shader)
    self.vao.activate()

  def update_buffers(self, shader):

    shader_change = self.shader_changed(shader)
    if shader_change:
      self.bind_buffers(shader)

    for b in self.opengl_buffers:
      data = getattr(self, b.surface_piece_attribute_name)
      if b.update_buffer_data(data) or shader_change:
        self.vao.bind_shader_variable(b)

  def get_elements(self):

    ta = self.triangles
    if ta is None:
      return None
    if self.display_style == self.Mesh:
      if self.masked_edges is None:
        from .._image3d import masked_edges
        self.masked_edges = (masked_edges(ta) if self.edge_mask is None
                             else masked_edges(ta, self.edge_mask))
      ta = self.masked_edges
    return ta
  elements = property(get_elements, None)

  def element_count(self):
    return self.elements.size

  def get_color(self):
    return self.color_rgba
  def set_color(self, rgba):
    self.color_rgba = rgba
    self.surface.redraw_needed = True
  color = property(get_color, set_color)
  '''Single color of surface piece used when per-vertex coloring is not specified.'''

  def opaque(self):
    return self.color_rgba[3] == 1 and (self.texture is None or self.opaque_texture)

  def draw(self, viewer):
    ''' Draw the surface piece.'''

    if self.triangles is None:
      return

    self.bind_buffers()     # Need bound vao to compile shader

    r = viewer.renderer()

    sopt = self.shader_options()
    p = r.use_shader(sopt)

    # Set color
    if self.vertex_colors is None and self.instance_colors is None:
      r.set_single_color(self.color_rgba)

    t = self.texture
    if not t is None:
      t.bind_texture()

    self.update_buffers(p)

    # Draw triangles
    eb = self.element_buffer
    etype = {self.Solid: eb.triangles,
             self.Mesh: eb.lines,
             self.Dot: eb.points}[self.display_style]
    eb.draw_elements(etype, self.instance_count())

    if not self.texture is None:
      self.texture.unbind_texture()

  def shader_options(self):
    sopt = {}
    from ..draw import Render as r
    lit = getattr(self, 'use_lighting', True)
    if not lit:
      sopt[r.SHADER_LIGHTING] = False
    if self.vertex_colors is None and self.instance_colors is None:
      sopt[r.SHADER_VERTEX_COLORS] = False
    t = self.texture
    if not t is None:
      sopt[r.SHADER_TEXTURE_2D] = True
      if hasattr(self, 'use_radial_warp') and self.use_radial_warp:
        sopt[r.SHADER_RADIAL_WARP] = True
    if not self.shift_and_scale is None:
      sopt[r.SHADER_SHIFT_AND_SCALE] = True
    elif not self.copies44 is None:
      sopt[r.SHADER_INSTANCING] = True
    return sopt

  def instance_count(self):
    if not self.shift_and_scale is None:
      ninst = len(self.shift_and_scale)
    elif not self.copies44 is None:
      ninst = len(self.copies44)
    else:
      ninst = None
    return ninst

  TRIANGLE_DISPLAY_MASK = 8
  EDGE0_DISPLAY_MASK = 1
  ALL_EDGES_DISPLAY_MASK = 7

  def get_triangle_and_edge_mask(self):
    return self.edge_mask
  def set_triangle_and_edge_mask(self, temask):
    self.edge_mask = temask
    self.surface.redraw_needed = True
  triangle_and_edge_mask = property(get_triangle_and_edge_mask,
                                    set_triangle_and_edge_mask)
  '''
  The triangle and edge mask is a 1-dimensional int32 numpy array of length equal
  to the number of triangles.  The lowest 4 bits are used to control display of
  the corresponding triangle and display of its 3 edges in mesh mode.
  '''
    
  def set_edge_mask(self, emask):

    em = self.edge_mask
    if em is None:
      if emask is None:
        return
      em = (emask & self.ALL_EDGES_DISPLAY_MASK)
      em |= self.TRIANGLE_DISPLAY_MASK
      self.edge_mask = em
    else:
      if emask is None:
        em |= self.ALL_EDGES_DISPLAY_MASK
      else:
        em = (em & self.TRIANGLE_DISPLAY_MASK) | (em & emask)
      self.edge_mask = em

    self.surface.redraw_needed = True
    self.masked_edges = None

  def bounds(self):
    '''
    Return the bounds of the surface piece in surface coordinates including
    any surface piece copies, but not including whole surface copies.
    '''
    # TODO: cache surface piece bounds
    va = self.vertices
    if va is None or len(va) == 0:
      return None
    xyz_min = va.min(axis = 0)
    xyz_max = va.max(axis = 0)
    sas = self.shift_and_scale
    if not sas is None and len(sas) > 0:
      xyz = sas[:,:3]
      xyz_min += xyz.min(axis = 0)
      xyz_max += xyz.max(axis = 0)
      # TODO: use scale factors
    b = (xyz_min, xyz_max)
    if self.copies:
      from ..geometry import bounds
      b = bounds.copies_bounding_box(b, self.copies)
    return b

  def first_intercept(self, mxyz1, mxyz2):
    '''
    Find the first intercept of a line segment with the surface piece and
    return the fraction of the distance along the segment where the intersection occurs
    or None if no intersection occurs.  Intercepts with masked triangle are included.
    '''
    # TODO check intercept of bounding box as optimization
    # TODO handle surface piece shift_and_scale.
    f = None
    va, ta = self.geometry
    from .. import _image3d
    if len(self.copies) == 0:
      fmin, tmin = _image3d.closest_geometry_intercept(va, ta, mxyz1, mxyz2)
      if not fmin is None and (f is None or fmin < f):
        f = fmin
    else:
      for tf in self.copies:
        cxyz1, cxyz2 = tf.inverse() * (mxyz1, mxyz2)
        fmin, tmin = _image3d.closest_geometry_intercept(va, ta, cxyz1, cxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
    return f

class Surface_Piece_Selection:
  '''
  Represent a selected surface piece as a generic selection object.
  '''
  def __init__(self, p):
    self.piece = p
  def description(self):
    p = self.piece
    n =  '%d triangles' % len(p.triangles) if p.name is None else p.name
    d = '%s %s' % (p.surface.name, n)
    return d
  def models(self):
    return [self.piece.surface]

def surface_image(qi, pos, size, surf = None):
  '''
  Make a new surface piece and texture map a QImage onto it.
  '''
  rgba = image_rgba_array(qi)
  if surf is None:
    surf = Surface('Image')
  return rgba_surface_piece(rgba, pos, size, surf)

def rgba_surface_piece(rgba, pos, size, surf):
  '''
  Make a new surface piece and texture map an RGBA color array onto it.
  '''
  p = surf.new_piece()
  x,y = pos
  sx,sy = size
  from numpy import array, float32, uint32
  vlist = array(((x,y,0),(x+sx,y,0),(x+sx,y+sy,0),(x,y+sy,0)), float32)
  tlist = array(((0,1,2),(0,2,3)), uint32)
  tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
  p.geometry = vlist, tlist
  p.color = (1,1,1,1)         # Modulates texture values
  p.use_lighting = False
  p.texture_coordinates = tc
  from ..draw import Texture
  p.texture = Texture(rgba)
  return p

# Extract rgba values from a QImage.
def image_rgba_array(i):
    s = i.size()
    w,h = s.width(), s.height()
    from ..ui.qt import QtGui
    i = i.convertToFormat(QtGui.QImage.Format_RGB32)    #0ffRRGGBB
    b = i.bits()        # sip.voidptr
    n = i.byteCount()
    import ctypes
    si = ctypes.string_at(int(b), n)
    # si = b.asstring(n)  # Uses METH_OLDARGS in SIP 4.10, unsupported in Python 3

    # TODO: Handle big-endian machine correctly.
    # Bytes are B,G,R,A on little-endian machine.
    from numpy import ndarray, uint8
    rgba = ndarray(shape = (h,w,4), dtype = uint8, buffer = si)

    # Flip vertical axis.
    rgba = rgba[::-1,:,:].copy()

    # Swap red and blue to get R,G,B,A
    t = rgba[:,:,0].copy()
    rgba[:,:,0] = rgba[:,:,2]
    rgba[:,:,2] = t

    return rgba

def opengl_matrices(places):
  '''
  Convert a list of Place objects into a numpy array of 3 by 4 matrices
  for creating surface piece instances.
  '''
  m34_list = tuple(p.matrix for p in places)
  n = len(m34_list)
  from numpy import empty, float32, transpose
  m = empty((n,4,4), float32)
  m[:,:,:3] = transpose(m34_list, axes = (0,2,1))
  m[:,:3,3] = 0
  m[:,3,3] = 1
  return m
