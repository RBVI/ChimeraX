'''
Drawing scene graph
===================
'''

class Drawing:
  '''
  A Drawing represents a tree of objects each consisting of a set of triangles in 3 dimensional space.
  Drawings are used to draw molecules, density maps, geometric shapes and other models.
  A Drawing has a name, a unique id number which is a positive integer, it can be displayed or hidden,
  has a placement in space, or multiple copies can be placed in space, and a drawing can be selected.
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
    self._displayed = True
    self._displayed_positions = None    # bool numpy array, show only some positions
    self._selected = False
    self._selected_positions = None     # bool numpy array, selected positions
    self._selected_triangles_mask = None # bool numpy array
    self._child_drawings = []

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
    self.use_lighting = True
    self.use_radial_warp = False

    # Derived arrays used for instancing
    self.instance_shift_and_scale = None    # N by 4 array, (x,y,z,scale)
    self.instance_matrices = None	    # 4x4 matrices for displayed instances
    self.instance_colors = None

    # OpenGL rendering                                    
    self.bindings = None                    # Holds the buffer pointers and shader variable bindings
    self.opengl_buffers = []
    self.elements = None                    # Triangles after mask applied
    self.element_buffer = None
    self.shader = None
    self.need_buffer_update = True
    self.reverse_order_children = False     # Used by grayscale rendering for depth ordering

  def __setattr__(self, key, value):
    if key in self.effects_shader:
      self.shader = None
    if key in self.effects_buffers:
      self.need_buffer_update = True
    super(Drawing,self).__setattr__(key, value)

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
    return self._displayed
  def set_display(self, display):
    self._displayed = display
    self.redraw_needed()
  display = property(get_display, set_display)
  '''Whether or not the surface is drawn.'''

  def display_positions(self, position_mask):
    self._displayed_positions = position_mask
    self.redraw_needed()

  def get_selected(self):
    return self._selected
  def set_selected(self, sel):
    self._selected = sel
    self.redraw_needed()
  selected = property(get_selected, set_selected)
  '''Whether or not the drawing is selected.'''

  def get_selected_positions(self):
    return self._selected_positions
  def set_selected_positions(self, spos):
    self._selected_positions = spos
    self.redraw_needed()
  selected_positions = property(get_selected_positions, set_selected_positions)
  '''Mask specifying which drawing positions are selected.'''

  def get_selected_triangles_mask(self):
    return self._selected_triangles_mask
  def set_selected_triangles_mask(self, tmask):
    self._selected_triangles_mask = tmask
    self.redraw_needed()
  selected_triangles_mask = property(get_selected_triangles_mask, set_selected_triangles_mask)
  '''Mask specifying which triangles are selected.'''

  def any_part_selected(self):
    if self._selected:
      return True
    sp = self._selected_positions
    if not sp is None and sp.sum() > 0:
      return True
    tmask = self._selected_triangles_mask
    if not tmask is None and tmask.sum() > 0:
      return True
    for d in self.child_drawings():
      if d.any_part_selected():
        return True
    return False

  def clear_selection(self):
    self.selected = False
    self.selected_positions = None

  def promote_selection(self):
    above = self.above_selected()
    if above:
      for d in above:
        d.selected = True
      if not hasattr(self, 'promotion_tower'):
        self.promotion_tower = [above]
      else:
        pt = self.promotion_tower
        for d in pt:
          if not d.selected:
            self.promotion_tower = pt = []
            break
        pt.append(above)

  def above_selected(self):
    if self.selected:
      return []
    if self.child_or_copies_selected():
      return [self]
    return sum((d.above_selected() for d in self.child_drawings()), [])

  def child_or_copies_selected(self):
    sp = self._selected_positions
    if not sp is None and sp.sum() > 0:
      return True
    stm = self._selected_triangles_mask
    if stm is None and stm.sum() > 0:
      return True
    for d in self.child_drawings():
      if d.selected:
        return True
    return False

  def demote_selection(self):
    pt = getattr(self, 'promotion_tower', None)
    if pt:
      for d in pt.pop():
        d.selected = False

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

  def get_color(self):
    return self._colors[0]
  def set_color(self, rgba):
    from numpy import empty, uint8
    c = empty((len(self._positions),4),uint8)
    c[:,:] = rgba
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

  def get_geometry(self):
    return self.vertices, self.triangles
  def set_geometry(self, g):
    self.vertices, self.triangles = g
    self.masked_edges = None
    self.edge_mask = None
    self.redraw_needed()
  geometry = property(get_geometry, set_geometry)
  '''Geometry is the array of vertices and array of triangles.'''

  def empty_drawing(self):
    return self.vertices is None

  OPAQUE_DRAW_PASS = 'opaque'
  TRANSPARENT_DRAW_PASS = 'transparent'
  TRANSPARENT_DEPTH_DRAW_PASS = 'transparent depth'
  SELECTION_DRAW_PASS = 'selection'

  def draw(self, renderer, place, draw_pass, selected_only = False):
    '''Draw this drawing and children using the given draw pass.'''

    if not self.display:
      return

    if not self.empty_drawing():
      self.draw_self(renderer, place, draw_pass, selected_only)

    if self.child_drawings():
      sp = self._selected_positions
      sel_only = selected_only and not self.selected
      for i,p in enumerate(self.positions):
        so = sel_only and (sp is None or not sp[i])
        pp = place if p.is_identity() else place*p
        self.draw_children(renderer, pp, draw_pass, so)

  def draw_self(self, renderer, place, draw_pass, selected_only = False):
    '''Draw this drawing without children using the given draw pass.'''

    if (selected_only and
        not self.selected and
        self._selected_positions is None and
        self._selected_triangles_mask is None):
      return

    if len(self.positions) == 1:
      p = self.position
      pp = place if p.is_identity() else place*p
    else:
      pp = place
    renderer.set_model_matrix(pp)

    if draw_pass == self.OPAQUE_DRAW_PASS:
      if self.opaque():
          self.draw_geometry(renderer, selected_only)
    elif draw_pass in (self.TRANSPARENT_DRAW_PASS, self.TRANSPARENT_DEPTH_DRAW_PASS):
      if not self.opaque():
        self.draw_geometry(renderer, selected_only)
    elif draw_pass == self.SELECTION_DRAW_PASS:
      # TODO: Avoid creating new bindings when drawing selection.
      self.shader = None        # Use outline drawing shader
      self.draw_geometry(renderer, selected_only)
      self.shader = None        # Restore regular shader

  def draw_children(self, renderer, place, draw_pass, selected_only = False):
    dlist = self.child_drawings()
    if self.reverse_order_children:
      dlist = dlist[::-1]
    for d in dlist:
      d.draw(renderer, place, draw_pass, selected_only)

  def draw_geometry(self, renderer, selected_only = False):
    ''' Draw the geometry.'''

    if self.triangles is None:
      return

    new_bindings = self.set_shader(renderer)

    # Set color
    if self.vertex_colors is None and len(self._colors) == 1:
      renderer.set_single_color([c/255.0 for c in self._colors[0]])

    t = self.texture
    if not t is None:
      t.bind_texture()

    if self.need_buffer_update or new_bindings:
      self.update_buffers(new_bindings, selected_only)

    # Draw triangles
    eb = self.element_buffer
    etype = element_type(self.display_style)
    ni = self.instance_count()
    if ni > 0:
      eb.draw_elements(etype, ni)

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
    Find the first intercept of a line segment with the drawing and its children.
    Return the fraction of the distance along the segment where the intersection occurs
    and a Picked_Drawing object for the intercepted piece.  For no intersection
    two None values are returned.  This routine is used for selecting objects, for
    identifying objects during mouse-over, and to determine the front-most point
    in the center of view to be used as the interactive center of rotation.
    '''
    f, dpchain = self.first_drawing_intercept(mxyz1, mxyz2, exclude)
    s = Picked_Drawing(dpchain) if dpchain else None
    return f, s

  def first_drawing_intercept(self, mxyz1, mxyz2, exclude = None):
    '''
    Find the first intercept of a line segment with the drawing or its descendants and
    return the fraction of the distance along the segment where the intersection occurs
    or None if no intersection occurs.  Also return a list of pairs of drawing and copy number
    descending to the intercepted child drawing.
    '''
    f = dpchain = None
    if self.display and (exclude is None or not hasattr(self,exclude)):
      if not self.empty_drawing():
        fmin,p = self.first_intercept_excluding_children(mxyz1, mxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
          dpchain = [(self,p)]
      cd = self.child_drawings()
      if cd:
        pos = [p.inverse()*(mxyz1,mxyz2) for p in self.positions]
        for d in cd:
          if d.display and (exclude is None or not hasattr(d,exclude)):
            for cp, (cxyz1,cxyz2) in enumerate(pos):
              fmin,dc = d.first_drawing_intercept(cxyz1, cxyz2, exclude)
              if not fmin is None and (f is None or fmin < f):
                f = fmin
                dpchain = [(self,cp)] + dc
    return f, dpchain

  def first_intercept_excluding_children(self, mxyz1, mxyz2):
    '''
    Find the first intercept of a line segment with the drawing and
    return the fraction of the distance along the segment where the intersection occurs
    or None if no intersection occurs.  Intercepts with masked triangle are included.
    Children drawings are not considered.
    '''
    # TODO check intercept of bounding box as optimization
    f = None
    p = None    # Position number
    if self.empty_drawing():
      return f
    va,ta = self.geometry
    from .. import _image3d
    if self.positions.is_identity():
      fmin, tmin = _image3d.closest_geometry_intercept(va, ta, mxyz1, mxyz2)
      if not fmin is None and (f is None or fmin < f):
        f = fmin
        p = 0
    else:
      # TODO: This will be very slow for large numbers of copies.
      dp = self._displayed_positions
      for c,tf in enumerate(self.positions):
        if dp is None or dp[c]:
          cxyz1, cxyz2 = tf.inverse() * (mxyz1, mxyz2)
          fmin, tmin = _image3d.closest_geometry_intercept(va, ta, cxyz1, cxyz2)
          if not fmin is None and (f is None or fmin < f):
            f = fmin
            p = c
    return f, p

  def delete(self):
    '''
    Delete drawing and all child drawings.
    '''
    self.delete_geometry()
    self.remove_all_drawings()
  
  def delete_geometry(self):
    '''Release all the arrays and graphics memory associated with the surface piece.'''
    self._positions = None
    self._colors = None
    self._displayed_positions = None
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
    for b in self.opengl_buffers:
      b.delete_buffer()

    self.bindings = None
    self.was_deleted = True

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

  effects_buffers = set(('vertices', 'normals', 'vertex_colors', 'texture_coordinates', 'elements',
                         '_displayed_positions', '_colors', '_positions'))

  def update_buffers(self, new_bindings, selected_only = False):
    if len(self.opengl_buffers) == 0 and not self.vertices is None:
      self.create_opengl_buffers()

    self.elements = self.masked_elements(selected_only)

    self.update_instance_arrays(selected_only)

    for b in self.opengl_buffers:
      data = getattr(self, b.buffer_attribute_name)
      if b.update_buffer_data(data) or new_bindings:
        self.bindings.bind_shader_variable(b)

    self.need_buffer_update = False

  def update_instance_arrays(self, selected_only = False):
    ic = self._colors
    sas = self.positions.shift_and_scale_array()
    if sas is None:
      self.instance_shift_and_scale = None
      if len(self.positions) == 1:
        self.instance_matrices = None
        self.instance_colors = None
      else:
        self.instance_matrices = self.positions.opengl_matrices()
        self.instance_colors = ic
    else:
      self.instance_matrices = None
      self.instance_shift_and_scale = sas
      self.instance_colors = ic

    dp = self._displayed_positions        # bool array
    if selected_only and not self.selected:
      sp = self._selected_positions
      if not sp is None:
        import numpy 
        dp = sp if dp is None else numpy.logical_and(dp, sp)
    if not dp is None:
      im = self.instance_matrices
      self.instance_matrices = im[dp,:,:] if not im is None else None
      self.instance_colors = ic[dp,:] if not ic is None else None
      self.instance_shift_and_scale = sas[dp,:] if not sas is None else None

  def masked_elements(self, selected_only = False):

    ta = self.triangles
    if ta is None:
      return None
    if selected_only and not self.selected:
      tmask = self._selected_triangles_mask
      if not tmask is None:
        ta = ta[tmask,:]
    if self.display_style == self.Mesh:
      me = self.masked_edges
      if me is None or selected_only:
        from .._image3d import masked_edges
        me = (masked_edges(ta) if self.edge_mask is None
              else masked_edges(ta, self.edge_mask))
        if not selected_only:
          self.masked_edges = me
      ta = me
    return ta

  def use_bindings(self):
    s = self.shader
    new_bindings = (self.bindings is None or
                    (s != self.bindings.shader and not s is None))
    if new_bindings:
      from . import opengl
      self.bindings = opengl.Bindings(s)
    self.bindings.activate()
    return new_bindings

  def set_shader(self, renderer):
    new_bindings = self.use_bindings()     # Need bound vao to compile shader
    if self.shader is None:
      sopt = self.shader_options()
      self.shader = renderer.shader(sopt)
      new_bindings = self.use_bindings()
    renderer.use_shader(self.shader)
    return new_bindings

  def shader_options(self):
    sopt = {}
    from .opengl import Render as r
    lit = self.use_lighting
    if not lit:
      sopt[r.SHADER_LIGHTING] = False
    if self.vertex_colors is None and len(self._colors) == 1:
      sopt[r.SHADER_VERTEX_COLORS] = False
    t = self.texture
    if not t is None:
      sopt[r.SHADER_TEXTURE_2D] = True
      if self.use_radial_warp:
        sopt[r.SHADER_RADIAL_WARP] = True
    if not self.positions.shift_and_scale_array() is None:
      sopt[r.SHADER_SHIFT_AND_SCALE] = True
    elif len(self.positions) > 1:
      sopt[r.SHADER_INSTANCING] = True
    return sopt

  effects_shader = set(('use_lighting', 'vertex_colors', '_colors', 'texture', 'use_radial_warp', '_positions'))

  def instance_count(self):
    im = self.instance_matrices
    isas = self.instance_colors
    if not im is None:
      ninst = len(im)
    elif not isas is None:
      ninst = len(isas)
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
  selected_only = (draw_pass == Drawing.SELECTION_DRAW_PASS)
  for d in drawings:
    d.draw(r, place, draw_pass, selected_only)

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
  draw_multiple(drawings, r, p, Drawing.SELECTION_DRAW_PASS)
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

class Picked_Drawing:
  '''
  Represent a drawing chosen with the mouse as a generic selection object.
  '''
  def __init__(self, drawing_chain):
    self.drawing_chain = drawing_chain
  def description(self):
    d,c = self.drawing_chain[-1]
    fields = []
    if not d.name is None:
      fields.append(d.name)
    if len(d.positions) > 1:
      fields.append('copy %d' % c)
    fields.append('triangles %d' % len(d.triangles))
    desc = ' '.join(fields)
    return desc
  def models(self):
    d = self.drawing_chain[0][0]
    return [d]
  def select(self, toggle = False):
    d,c = self.drawing_chain[-1]
    n = len(d.positions)
    if n == 1:
      d.selected = not d.selected if toggle else True
    else:
      # Set selected position
      pmask = d.selected_positions
      if pmask is None:
        from numpy import zeros, bool
        pmask = zeros((n,), bool)
      pmask[c] = not pmask[c] if toggle else 1
      d.selected_positions = pmask

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
