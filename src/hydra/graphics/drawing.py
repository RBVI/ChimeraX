'''
Drawing scene graph
===================
'''

class Drawing:
  '''
  A Drawing represents a tree of objects each consisting of a set of triangles in 3 dimensional space.
  Drawings are used to draw molecules, density maps, geometric shapes and other models.
  A Drawing has a name, a unique id number which is a positive integer, it can be displayed or hidden,
  has a placement in space, or multiple copies can be placed in space, and a surface can be selected.
  The coordinates, colors, normal vectors and other geometric and display properties are managed by the
  Drawing objects.  Individual child drawings can be added or removed.  The purpose of child drawings is
  for convenience in adding and removing parts of a Drawing.

  Child drawings are created by the new_drawing() method and represent a set of triangles
  that can be added or removed from a parent drawing independent of other sets of triangles.  The basic
  data defining the triangles is an N by 3 array of vertices (float32 numpy array) and an array
  that defines triangles as 3 integer index values into the vertex array which define the
  3 corners of a triangle.  The triangle array is of shape T by 3 and is a numpy int32 array.
  The filled triangles or a mesh consisting of just the triangle edges can be shown.
  The vertices can be individually colored with linear interpolation of colors across triangles,
  or all triangles can be given the same color, or a 2-dimensional texture can be used to color
  the triangles with texture coordinates assigned to the vertices.  Transparency values can be assigned
  to the vertices. Individual triangles or triangle edges in mesh display style can be hidden.
  An N by 3 float array gives normal vectors, one normal per vertex, for lighting calculations.
  Multiple copies of the drawing be drawn with each specified
  by a position and orientation.  Copies can alternatively be specified by a shift and scale factor
  but no rotation, useful for copies of spheres.  Each copy can have its own single color, or all
  copies can use the same per-vertex or texture coloring.  Rendering of drawings is done with OpenGL.
  '''

  def __init__(self, name):
    self.name = name
    self.id = None                      # positive integer
    from ..geometry.place import Places
    self._positions = Places()          # Copies of drawing are placed at these positions
    self._colors = [(178,178,178,255)]  # Colors for each position, N by 4 uint8 numpy array
    self._child_drawings = []
    self.subsets = set(['displayed'])   # Subsets this drawing belongs to.

    self.redraw_needed = redraw_no_op
    self.was_deleted = False

    # Geometry and colors
    self.vertices = None
    self.triangles = None
    self.normals = None
    self.vertex_colors = None           # N by 4 uint8 values
    self.edge_mask = None
    self.masked_edges = None
    self.display_style = self.Solid
    self.texture = None
    self.texture_coordinates = None
    self.opaque_texture = False

    # Instancing
    # TODO: Generalize to instance subset.
    self.instance_display = None            # bool numpy array, show only some instances
    self.instance_shift_and_scale = None    # N by 4 array, (x,y,z,scale)
    self.instance_matrices = None	    # 4x4 matrices for displayed instances
    self.instance_colors = None

    # OpenGL rendering                                    
    self.bindings = None                # Holds the buffer pointers and shader variable bindings
    self.opengl_buffers = []
    self.element_buffer = None

  # Display styles
  Solid = 'solid'
  Mesh = 'mesh'
  Dot = 'dot'

  def child_drawings(self):
    '''Return the list of surface pieces.'''
    return self._child_drawings

  def all_drawings(self):
    '''Return all drawings including self and children at all levels.'''
    dlist = [self]
    for d in self.child_drawings():
      dlist.extend(d.all_drawings())
    return dlist

  def new_drawing(self, name = None):
    '''Create a new empty child drawing.'''
    d = Drawing(name)
    self.add_drawing(d)
    return d

  def add_drawing(self, d):
    '''Add a child drawing.'''
    d.redraw_needed = self.redraw_needed
    cd = self._child_drawings
    cd.append(d)
    d.id = len(cd)
    d.parent = self
    self.redraw_needed()

  def remove_drawing(self, d):
    '''Delete a specified child drawing.'''
    self._child_drawings.remove(d)
    d.delete()
    self.redraw_needed()

  def remove_drawings(self, drawings):
    '''Delete specified child drawings.'''
    dset = set(drawings)
    self._child_drawings = [d for d in self._child_drawings if not d in dset]
    for d in drawings:
      d.delete()
    self.redraw_needed()

  def remove_all_drawings(self):
    '''Delete all child drawings.'''
    self.remove_drawings(self.child_drawings())

  def set_redraw_callback(self, redraw_needed):
    self.redraw_needed = redraw_needed
    for d in self.child_drawings():
      d.set_redraw_callback(redraw_needed)

  def get_display(self):
    return 'displayed' in self.subsets
  def set_display(self, display):
    s = self.subsets
    d = 'displayed' in s
    if display != d:
      if display:
        s.add('displayed')
      else:
        s.remove('displayed')
      self.redraw_needed()
  display = property(get_display, set_display)
  '''Whether or not the surface is drawn.'''

  def get_selected(self):
    return 'selected' in self.subsets
  def set_selected(self, sel):
    s = self.subsets
    if sel:
      s.add('selected')
    else:
      s.discard('selected')
  selected = property(get_selected, set_selected)
  '''Whether or not the drawing is selected.'''

  def get_position(self):
    return self._positions[0]
  def set_position(self, pos):
    from ..geometry.place import Places
    self._positions = Places([pos])
    self.redraw_needed()
  position = property(get_position, set_position)
  '''Position and orientation of the surface in space.'''

  def get_positions(self):
    return self._positions
  def set_positions(self, positions):
    self._positions = positions
    self.redraw_needed()
  positions = property(get_positions, set_positions)
  '''
  Copies of the surface piece are placed using a 3 by 4 matrix with the first 3 columns
  giving a linear transformation, and the last column specifying a shift.
  '''

  def empty_drawing(self):
    return self.vertices is None

  def in_subset(self, only):
    self_in = children_in = True
    children_only = only
    s = self.subsets
    for o in only:
      require_in_parents = (o == 'displayed')
      if o not in s:
        self_in = False
        if require_in_parents:
          children_in = False
          break
      elif not require_in_parents:
        if children_only is only:
          children_only = set(only)
        children_only.remove(o)
    return self_in, children_in, children_only

  OPAQUE_DRAW_PASS = 'opaque'
  TRANSPARENT_DRAW_PASS = 'transparent'
  TRANSPARENT_DEPTH_DRAW_PASS = 'transparent depth'

  def draw(self, renderer, place, draw_pass, only = ['displayed'], reverse_order = False):
    '''Draw this drawing and children using the given draw pass.'''
    dself, dchildren, onlyc = self.in_subset(only)
    if not dself and not dchildren:
      return

    if dself and not self.empty_drawing():
      if len(self.positions) == 1:
        p = self.position
        pp = place if p.is_identity() else place*p
      else:
        pp = place
      renderer.set_model_matrix(pp)
      self.draw_self(renderer, draw_pass)

    if dchildren and self.child_drawings():
      for p in self.positions:
        pp = place if p.is_identity() else place*p
        self.draw_children(renderer, pp, draw_pass, onlyc, reverse_order)

  def draw_self(self, renderer, draw_pass):
    '''Draw this drawing without children using the given draw pass.'''
    if draw_pass == self.OPAQUE_DRAW_PASS:
      if self.opaque():
          self.draw_geometry(renderer)
    elif draw_pass in (self.TRANSPARENT_DRAW_PASS, self.TRANSPARENT_DEPTH_DRAW_PASS):
      if not self.opaque():
        self.draw_geometry(renderer)

  def draw_children(self, renderer, place, draw_pass, only = ['displayed'], reverse_order = False):
    dlist = self.child_drawings()
    if reverse_order:
      dlist = dlist[::-1]
    for d in dlist:
      d.draw(renderer, place, draw_pass, only, reverse_order)

  def draw_geometry(self, renderer):
    ''' Draw the geometry.'''

    if self.triangles is None:
      return

    self.bind_buffers()     # Need bound vao to compile shader

    # TODO: Optimize so shader options are not recomputed every frame.
    sopt = self.shader_options()
    p = renderer.use_shader(sopt)

    # Set color
    if self.vertex_colors is None and len(self._colors) == 1:
      renderer.set_single_color([c/255.0 for c in self._colors[0]])

    t = self.texture
    if not t is None:
      t.bind_texture()

    # TODO: Optimize so buffer update is not done if nothing changed.
    self.update_buffers(p)

    # Draw triangles
    eb = self.element_buffer
    etype = element_type(self.display_style)
    eb.draw_elements(etype, self.instance_count())

    if not self.texture is None:
      self.texture.unbind_texture()

  def bounds(self, positions = True):
    '''
    The bounds of drawing and children including undisplayed and all positions.
    '''
    # TODO: Should this only include displayed drawings?

    dbounds = [d.bounds() for d in self.child_drawings()]
    if not self.empty_drawing():
      dbounds.append(self.geometry_bounds())
    from ..geometry import bounds
    b = bounds.union_bounds(dbounds)
    if positions:
      b = bounds.copies_bounding_box(b, self.positions)
    return b

  def geometry_bounds(self):
    '''
    Return the bounds of the drawing not including positions nor children.
    '''
    # TODO: cache bounds
    va = self.vertices
    if va is None or len(va) == 0:
      return None
    xyz_min = va.min(axis = 0)
    xyz_max = va.max(axis = 0)
    sas = self.positions.shift_and_scale_array()
    if not sas is None and len(sas) > 0:
      xyz = sas[:,:3]
      xyz_min += xyz.min(axis = 0)
      xyz_max += xyz.max(axis = 0)
      # TODO: use scale factors
    b = (xyz_min, xyz_max)
    return b

  def first_intercept(self, mxyz1, mxyz2, exclude = None):
    '''
    Find the first intercept of a line segment with the displayed pieces of the surface.
    Return the fraction of the distance along the segment where the intersection occurs
    and a Drawing_Selection object for the intercepted piece.  For no intersection
    two None values are returned.  This routine is used to determine the front-most point
    in the center of view to be used as the interactive center of rotation.
    '''
    f = None
    sd = None
    from .. import _image3d
    for d in self.all_drawings():
      if d.display and (exclude is None or not hasattr(d,exclude)):
        fmin = d.first_geometry_intercept(mxyz1, mxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
          sd = d
    return f, Drawing_Selection(sd)

  def first_geometry_intercept(self, mxyz1, mxyz2):
    '''
    Find the first intercept of a line segment with the surface piece and
    return the fraction of the distance along the segment where the intersection occurs
    or None if no intersection occurs.  Intercepts with masked triangle are included.
    Children drawings are not considered.
    '''
    if self.empty_drawing():
      return None
    # TODO check intercept of bounding box as optimization
    f = None
    va, ta = self.geometry
    from .. import _image3d
    if self.positions.is_identity():
      fmin, tmin = _image3d.closest_geometry_intercept(va, ta, mxyz1, mxyz2)
      if not fmin is None and (f is None or fmin < f):
        f = fmin
    else:
      # TODO: This will be very slow for large numbers of copies.
      id = self.instance_display
      for c,tf in enumerate(self.positions):
        if id is None or id[c]:
          cxyz1, cxyz2 = tf.inverse() * (mxyz1, mxyz2)
          fmin, tmin = _image3d.closest_geometry_intercept(va, ta, cxyz1, cxyz2)
          if not fmin is None and (f is None or fmin < f):
            f = fmin
    return f

  def delete(self):
    '''
    Delete drawing and all child drawings.
    '''
    self.delete_geometry()
    self.remove_all_drawings()
  
  def delete_geometry(self):
    '''Release all the arrays and graphics memory associated with the surface piece.'''
    self.vertices = None
    self.triangles = None
    self.normals = None
    self.edge_mask = None
    self.texture = None
    self.texture_coordinates = None
    self.masked_edges = None
    self.instance_shift_and_scale = None
    self.instance_matrices = None
    self.instance_colors = None
    self.instance_display = None
    for b in self.opengl_buffers:
      b.delete_buffer()

    self.bindings = None
    self.was_deleted = True

  def get_geometry(self):
    return self.vertices, self.triangles
  def set_geometry(self, g):
    self.vertices, self.triangles = g
    self.masked_edges = None
    self.edge_mask = None
    self.redraw_needed()
  geometry = property(get_geometry, set_geometry)
  '''Geometry is the array of vertices and array of triangles.'''

  def shader_changed(self, shader):
    return self.bindings is None or (shader != self.bindings.shader and not shader is None)

  def bind_buffers(self, shader = None):
    if self.shader_changed(shader):
      from . import opengl
      self.bindings = opengl.Bindings(shader)
    self.bindings.activate()

  def create_opengl_buffers(self):
    # Surface piece attribute name, shader variable name, instancing
    from . import opengl
    from numpy import uint32, uint8
    bufs = (('vertices', opengl.VERTEX_BUFFER),
            ('normals', opengl.NORMAL_BUFFER),
            ('vertex_colors', opengl.VERTEX_COLOR_BUFFER),
            ('texture_coordinates', opengl.TEXTURE_COORDS_2D_BUFFER),
            ('elements', opengl.ELEMENT_BUFFER),
            ('instance_shift_and_scale', opengl.INSTANCE_SHIFT_AND_SCALE_BUFFER),
            ('instance_matrices', opengl.INSTANCE_MATRIX_BUFFER),
            ('instance_colors', opengl.INSTANCE_COLOR_BUFFER),
            )
    obufs = []
    for a,v in bufs:
      b = opengl.Buffer(v)
      b.buffer_attribute_name = a
      obufs.append(b)
      if a == 'elements':
        self.element_buffer = b
    self.opengl_buffers = obufs

  def update_buffers(self, shader):
    if len(self.opengl_buffers) == 0 and not self.vertices is None:
      self.create_opengl_buffers()

    shader_change = self.shader_changed(shader)
    if shader_change:
      self.bind_buffers(shader)

    disp = self.instance_display        # bool array
    ic = self._colors
    sas = self.positions.shift_and_scale_array()
    if sas is None:
      if len(self.positions) == 1:
        self.instance_matrices = None
        self.instance_colors = None
      elif disp is None:
          self.instance_matrices = self.positions.opengl_matrices()
          self.instance_colors = ic
      else:
        # TODO: Changing instance_display does not cause update.
        self.instance_matrices = self.positions.opengl_matrices()[disp,:,:]
        self.instance_colors = ic[disp,:] if not ic is None else None
    elif disp is None:
      self.instance_shift_and_scale = sas if disp is None else sas[disp,:]
      self.instance_colors = ic
    else:
      self.instance_shift_and_scale = sas[disp,:]
      self.instance_colors = ic[disp,:] if not ic is None else None

    for b in self.opengl_buffers:
      data = getattr(self, b.buffer_attribute_name)
      if b.update_buffer_data(data) or shader_change:
        self.bindings.bind_shader_variable(b)

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
    return self._colors[0]
  def set_color(self, rgba):
    from numpy import empty, uint8
    c = empty((1,4),uint8)
    c[0,:] = rgba
    self._colors = c
    self.redraw_needed()
  color = property(get_color, set_color)
  '''Single color of drawing used when per-vertex coloring is not specified.'''

  def get_colors(self):
    return self._colors
  def set_colors(self, rgba):
    self._colors = rgba
    self.redraw_needed()
  colors = property(get_colors, set_colors)
  '''Color for each position used when per-vertex coloring is not specified.'''

  def opaque(self):
    # TODO: Should render transparency for each copy separately
    return self._colors[0][3] == 255 and (self.texture is None or self.opaque_texture)

  def showing_transparent(self):
    '''Are any transparent objects being displayed. Includes all children.'''
    if self.display:
      if not self.empty_drawing() and not self.opaque():
        return True
      for d in self.child_drawings():
        if d.showing_transparent():
          return True
    return False

  def shader_options(self):
    sopt = {}
    from ..graphics import Render as r
    lit = getattr(self, 'use_lighting', True)
    if not lit:
      sopt[r.SHADER_LIGHTING] = False
    if self.vertex_colors is None and len(self._colors) == 1:
      sopt[r.SHADER_VERTEX_COLORS] = False
    t = self.texture
    if not t is None:
      sopt[r.SHADER_TEXTURE_2D] = True
      if hasattr(self, 'use_radial_warp') and self.use_radial_warp:
        sopt[r.SHADER_RADIAL_WARP] = True
    if not self.positions.shift_and_scale_array() is None:
      sopt[r.SHADER_SHIFT_AND_SCALE] = True
    elif len(self.positions) > 1:
      sopt[r.SHADER_INSTANCING] = True
    return sopt

  def instance_count(self):
    if not self.instance_display is None:
      ninst = self.instance_display.sum()
    else:
      ninst = len(self.positions)
    return ninst

  TRIANGLE_DISPLAY_MASK = 8
  EDGE0_DISPLAY_MASK = 1
  ALL_EDGES_DISPLAY_MASK = 7

  def get_triangle_and_edge_mask(self):
    return self.edge_mask
  def set_triangle_and_edge_mask(self, temask):
    self.edge_mask = temask
    self.redraw_needed()
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

    self.redraw_needed()
    self.masked_edges = None

def draw_drawings(renderer, cvinv, drawings):
  r = renderer
  r.set_view_matrix(cvinv)
  from ..geometry.place import Place
  p = Place()
  draw_multiple(drawings, r, p, Drawing.OPAQUE_DRAW_PASS)
  if any_transparent_drawings(drawings):
    r.draw_transparent(lambda: draw_multiple(drawings, r, p, Drawing.TRANSPARENT_DEPTH_DRAW_PASS),
                       lambda: draw_multiple(drawings, r, p, Drawing.TRANSPARENT_DRAW_PASS))

def draw_multiple(drawings, r, place, draw_pass):
  for d in drawings:
    d.draw(r, place, draw_pass)

def any_transparent_drawings(drawings):
  for d in drawings:
    if d.showing_transparent():
      return True
  return False

def draw_overlays(drawings, renderer):

  r = renderer
  r.set_projection_matrix(((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)))
  from ..geometry import place
  p0 = place.identity()
  r.set_view_matrix(p0)
  r.set_model_matrix(p0)
  r.enable_depth_test(False)
  draw_multiple(drawings, r, p0, Drawing.OPAQUE_DRAW_PASS)
  r.enable_blending(True)
  draw_multiple(drawings, r, p0, Drawing.TRANSPARENT_DRAW_PASS)
  r.enable_depth_test(True)

def draw_outline(window_size, renderer, cvinv, drawings):
  r = renderer
  r.set_view_matrix(cvinv)
  r.start_rendering_outline(window_size)
  from ..geometry.place import Place
  p = Place()
  draw_multiple(drawings, r, p, Drawing.OPAQUE_DRAW_PASS)
  draw_multiple(drawings, r, p, Drawing.TRANSPARENT_DRAW_PASS)
  r.finish_rendering_outline()

def element_type(display_style):
  from .opengl import Buffer
  if display_style == Drawing.Solid:
    t = Buffer.triangles
  elif display_style == Drawing.Mesh:
    t = Buffer.lines
  elif display_style == Drawing.Dot:
    t = Buffer.points
  return t

def redraw_no_op():
  pass

class Drawing_Selection:
  '''
  Represent a selected drawing as a generic selection object.
  '''
  def __init__(self, d):
    self.drawing = d
  def description(self):
    d = self.drawing
    nt =  '%d triangles' % len(d.triangles)
    desc = nt if d.name is None else ('%s %s' % (d.name, nt))
    return desc
  def models(self):
    d = self.drawing
    while hasattr(d, 'parent'):
      d = d.parent
    return [d]

def image_drawing(qi, pos, size, drawing = None):
  '''
  Make a new surface piece and texture map a QImage onto it.
  '''
  rgba = image_rgba_array(qi)
  if drawing is None:
    drawing = Drawing('Image')
  return rgba_drawing(rgba, pos, size, drawing)

def rgba_drawing(rgba, pos, size, drawing):
  '''
  Make a new surface piece and texture map an RGBA color array onto it.
  '''
  d = drawing.new_drawing()
  x,y = pos
  sx,sy = size
  from numpy import array, float32, uint32
  vlist = array(((x,y,0),(x+sx,y,0),(x+sx,y+sy,0),(x,y+sy,0)), float32)
  tlist = array(((0,1,2),(0,2,3)), uint32)
  tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
  d.geometry = vlist, tlist
  d.color = (255,255,255,255)         # Modulates texture values
  d.use_lighting = False
  d.texture_coordinates = tc
  from . import opengl
  d.texture = opengl.Texture(rgba)
  return d

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
