from OpenGL import GL

class Surface:

  def __init__(self):
    self.id = 0
    self.displayed = True
    self.placement = ((1,0,0,0),(0,1,0,0),(0,0,1,0))
    self.copies = []
    self.plist = []
    self.selected = False
    self.redraw_needed = False

  def surface_pieces(self):
    return self.plist

  def newPiece(self):
    p = Surface_Piece()
    p.surface = self
    self.plist.append(p)
    self.redraw_needed = True
    return p

  def removePiece(self, p):
    self.plist.remove(p)
    p.delete()
    self.redraw_needed = True

  def removePieces(self, pieces):
    pset = set(pieces)
    self.plist = [p for p in self.plist if not p in pset]
    for p in pieces:
      p.delete()
    self.redraw_needed = True

  def removeAllPieces(self):
    self.removePieces(self.plist)

  def get_display(self):
    return self.displayed
  def set_display(self, display):
    self.displayed = display
    self.redraw_needed = True
  display = property(get_display, set_display)

  def get_place(self):
    return self.placement
  def set_place(self, place):
    self.placement = place
    self.redraw_needed = True
  place = property(get_place, set_place)

  def showing_transparent(self):
    for p in self.plist:
      if p.display and not p.opaque():
        return True
    return False

  def draw(self, viewer, draw_pass, reverse_order = False):
    plist = self.plist[::-1] if reverse_order else self.plist
    self.draw_pieces(plist, viewer, draw_pass)

  def draw_pieces(self, plist, viewer, draw_pass):

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
    return union_bounds(p.bounds() for p in self.plist)

  def placed_bounds(self):
    b = self.bounds()
    if b is None:
      return b
    if self.copies:
      copies = self.copies
    elif self.placement != ((1,0,0,0),(0,1,0,0),(0,0,1,0)):
      copies = [self.placement]
    else:
      return b
    return copies_bounding_box(b, copies)

  def first_intercept(self, mxyz1, mxyz2):
    f = None
    # TODO handle surface model copies.
    from . import _image3d
    for p in self.plist:
      if p.display:
        fmin = p.first_intercept(mxyz1, mxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
    return f

  def delete(self):
    self.removeAllPieces()
  
class Surface_Piece(object):

  Solid = 'solid'
  Mesh = 'mesh'
  Dot = 'dot'

  TRIANGLE_DISPLAY_MASK = 8
  EDGE0_DISPLAY_MASK = 1
  ALL_EDGES_DISPLAY_MASK = 7

  def __init__(self):
    self.vertices = None
    self.triangles = None
    self.normals = None
    self.shift_and_scale = None         # Instance copies
    self.copies34 = []                  # Instance matrices, 3x4
    self.copies44 = None                # Instance matrices, 4x4 opengl
    self.vertex_colors = None
    self.instance_colors = None
    self.edge_mask = None
    self.masked_edges = None
    self.display = True
    self.displayStyle = self.Solid
    self.color_rgba = (.7,.7,.7,1)
    self.textureId = None
    self.textureFree = True             # Does Surface delete the texture
    self.textureCoordinates = None
    self.opaqueTexture = False
    self.__destroyed__ = False

    self.vao = None     	# Holds the buffer pointers and bindings
    self.shader = None		# VAO bindings are for this shader

    # Surface piece attribute name, shader variable name, instancing
    from numpy import uint32, uint8
    bufs = (('vertices', 'position', {}),
            ('normals', 'normal', {}),
            ('shift_and_scale',  'instanceShiftAndScale', {'instance_buffer':True}),
            ('copies44', 'instancePlacement', {'instance_buffer': True}),
            ('vertex_colors',  'vcolor', {'value_type':uint8, 'normalize':True}),
            ('instance_colors',  'vcolor', {'instance_buffer':True, 'value_type':uint8, 'normalize':True}),
            ('textureCoordinates', 'tex_coord_2d', {}),
            ('elements', None, {'buffer_type':GL.GL_ELEMENT_ARRAY_BUFFER, 'value_type':uint32}),
            )
    obufs = []
    for a,v,kw in bufs:
      b = OpenGL_Buffer(v,**kw)
      b.surface_piece_attribute_name = a
      obufs.append(b)
    self.opengl_buffers = obufs
    self.element_buffer = obufs[-1]

  def delete(self):

    self.vertices = None
    self.triangles = None
    self.normals = None
    self.edge_mask = None
    self.textureCoordinates = None
    self.masked_edges = None
    for b in self.opengl_buffers:
      b.delete_buffer()

    if not self.textureId is None and self.textureFree:
      GL.glDeleteTextures((self.textureId,))
    self.textureId = None

    if not self.vao is None:
      GL.glDeleteVertexArrays(1, (self.vao,))
    self.vao = None

  def get_geometry(self):
    return self.vertices, self.triangles
  def set_geometry(self, g):
    self.vertices, self.triangles = g
    self.masked_edges = None
    self.edge_mask = None
    self.surface.redraw_needed = True
  geometry = property(get_geometry, set_geometry)

  def get_copies(self):
    return self.copies34
  def set_copies(self, copies):
    self.copies34 = copies
    self.copies44 = opengl_matrices(copies) if copies else None
    self.surface.redraw_needed = True
  copies = property(get_copies, set_copies)

  def new_vertex_array_object(self):
    if not self.vao is None:
      GL.glDeleteVertexArrays(1, (self.vao,))
    self.vao = GL.glGenVertexArrays(1)

  def bind_vertex_array_object(self):
    if self.vao is None:
      self.vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(self.vao)

  def update_buffers(self, shader):

    shader_changed = (shader != self.shader)
    if shader_changed:
      self.shader = shader
      self.new_vertex_array_object()
      self.bind_vertex_array_object()

    for b in self.opengl_buffers:
      data = getattr(self, b.surface_piece_attribute_name)
      b.update_buffer(data, shader, shader_changed)

  def get_elements(self):

    ta = self.triangles
    if ta is None:
      return None
    if self.displayStyle == self.Mesh:
      if self.masked_edges is None:
        from ._image3d import masked_edges
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

  def opaque(self):
    return self.color_rgba[3] == 1 and (self.textureId is None or self.opaqueTexture)

  def draw(self, viewer):

    if self.triangles is None:
      return

    self.bind_vertex_array_object()     # Need bound vao to compile shader

    p = self.set_shader(viewer)

    self.update_buffers(p)

    # Set color
    if self.instance_colors is None:
      GL.glVertexAttrib4f(p.attribute_id("vcolor"), *self.color_rgba)

    # Draw triangles
    etype = {self.Solid:GL.GL_TRIANGLES,
             self.Mesh:GL.GL_LINES,
             self.Dot:GL.GL_POINTS}[self.displayStyle]
    self.element_buffer.draw_elements(etype, self.instance_count())

    if not self.textureId is None:
      GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

  def set_shader(self, viewer):
    # Push special shader if needed.
    skw = {}
    lit = getattr(self, 'useLighting', True)
    if not lit:
      skw['lighting'] = False
    t = self.textureId
    if not t is None:
      GL.glBindTexture(GL.GL_TEXTURE_2D, t)
      skw['texture2d'] = True
    if not self.shift_and_scale is None:
      skw['shiftAndScale'] = True
    elif not self.copies44 is None:
      skw['instancing'] = True
#    if viewer.selected and not self.surface.selected:
#      skw['unselected'] = True
    if self.surface.selected:
      skw['selected'] = True
    p = viewer.set_shader(**skw)
    return p

  def instance_count(self):
    if not self.shift_and_scale is None:
      ninst = len(self.shift_and_scale)
    elif len(self.copies) > 0:
      ninst = len(self.copies)
    else:
      ninst = None
    return ninst

  def get_triangle_and_edge_mask(self):
    return edge_mask
  def set_triangle_and_edge_mask(self, temask):
    self.edge_mask = temask
    self.surface.redraw_needed = True
  triangleAndEdgeMask = property(get_triangle_and_edge_mask,
                                 set_triangle_and_edge_mask)
    
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
      b = copies_bounding_box(b, self.copies)
    return b

  def first_intercept(self, mxyz1, mxyz2):
    # TODO check intercept of bounding box as optimization
    # TODO handle surface piece shift_and_scale.
    f = None
    va, ta = self.geometry
    from . import _image3d
    if len(self.copies) == 0:
      fmin, tmin = _image3d.closest_geometry_intercept(va, ta, mxyz1, mxyz2)
      if not fmin is None and (f is None or fmin < f):
        f = fmin
    else:
      from . import matrix
      for tf in self.copies:
        cxyz1, cxyz2 = matrix.apply_matrix(matrix.invert_matrix(tf), (mxyz1, mxyz2))
        fmin, tmin = _image3d.closest_geometry_intercept(va, ta, cxyz1, cxyz2)
        if not fmin is None and (f is None or fmin < f):
          f = fmin
    return f

class OpenGL_Buffer:

  from numpy import float32
  def __init__(self, shader_variable_name,
               buffer_type = GL.GL_ARRAY_BUFFER, value_type = float32,
               normalize = False, instance_buffer = False):

    self.shader_variable_name = shader_variable_name
    self.opengl_buffer = None
    self.buffered_array = None  # numpy array for vbo
    self.buffered_data = None   # data need not be numpy array
    self.value_type = value_type
    self.buffer_type = buffer_type
    self.normalize = normalize
    self.instance_buffer = instance_buffer
    self.bound_attr_ids = []

  def update_buffer_data(self, data):

    if data is self.buffered_data:
      return False

    self.delete_buffer()

    if not data is None:
      b = GL.glGenBuffers(1);
      btype = self.buffer_type
      GL.glBindBuffer(btype, b)
      d = data if data.dtype == self.value_type else data.astype(self.value_type)
      size = d.size * d.itemsize        # Bytes
      GL.glBufferData(btype, size, d, GL.GL_STATIC_DRAW)
      GL.glBindBuffer(btype, 0)
      self.opengl_buffer = b
      self.buffered_array = d
      self.buffered_data = data

    return True

  def delete_buffer(self):

    if self.opengl_buffer is None:
      return
    GL.glDeleteBuffers(1, [self.opengl_buffer])
    self.opengl_buffer = None
    self.buffered_array = None
    self.buffered_data = None

  def bind_shader_variable(self, shader):

    buf_id = self.opengl_buffer
    if buf_id is None:
      # Unbind already bound variable
      for a in self.bound_attr_ids:
        GL.glDisableVertexAttribArray(a)
      self.bound_attr_ids = []
      if self.buffer_type == GL.GL_ELEMENT_ARRAY_BUFFER:
        GL.glBindBuffer(self.buffer_type, 0)
      return

    vname = self.shader_variable_name
    if vname is None:
      if self.buffer_type == GL.GL_ELEMENT_ARRAY_BUFFER:
        # Element array buffer binding is saved in VAO.
        GL.glBindBuffer(self.buffer_type, buf_id)
      return

    attr_id = shader.attribute_id(vname)
    if attr_id == -1:
      raise NameError('Failed to find shader attribute %s\nshader capabilites = %s'
                      % (self.shader_variable_name, str(shader.capabilities)))
    nattr = self.attribute_count()
    ncomp = self.component_count()
    from numpy import float32, uint8
    gtype = {float32:GL.GL_FLOAT,
             uint8:GL.GL_UNSIGNED_BYTE}[self.value_type]
    btype = self.buffer_type
    normalize = GL.GL_TRUE if self.normalize else GL.GL_FALSE

    GL.glBindBuffer(btype, buf_id)
    if nattr == 1:
      GL.glVertexAttribPointer(attr_id, ncomp, gtype, normalize, 0, None)
      GL.glEnableVertexAttribArray(attr_id)
      glVertexAttribDivisor(attr_id, 1 if self.instance_buffer else 0)
      self.bound_attr_ids = [attr_id]
    else:
      # Matrices use multiple vector attributes
      esize = self.array_element_bytes()
      abytes = ncomp * esize
      stride = nattr * abytes
      import ctypes
      for a in range(nattr):
        # Pointer arg must be void_p, not an integer.
        p = ctypes.c_void_p(a*abytes)
        GL.glVertexAttribPointer(attr_id+a, ncomp, gtype, normalize, stride, p)
        GL.glEnableVertexAttribArray(attr_id+a)
        glVertexAttribDivisor(attr_id+a, 1 if self.instance_buffer else 0)
      self.bound_attr_ids = [attr_id+a for a in range(nattr)]
    GL.glBindBuffer(btype, 0)

    return attr_id

  def attribute_count(self):
    # matrix attributes use multiple attribute ids
    barray = self.buffered_array
    if barray is None:
      return 0
    bshape = barray.shape
    nattr = 1 if len(bshape) == 2 else bshape[1]
    return nattr

  def component_count(self):
    barray = self.buffered_array
    if barray is None:
      return 0
    ncomp = barray.shape[-1]
    return ncomp
  
  def array_element_bytes(self):
    barray = self.buffered_array
    return 0 if barray is None else barray.itemsize

  def update_buffer(self, data, shader, new_vao):

    if new_vao:
      self.bound_attr_ids = []

    if self.update_buffer_data(data) or new_vao:
      self.bind_shader_variable(shader)

  def draw_elements(self, element_type = GL.GL_TRIANGLES, ninst = None):

    # Don't bind element buffer since it is bound by VAO.
    ne = self.buffered_array.size
    if ninst is None:
      GL.glDrawElements(element_type, ne, GL.GL_UNSIGNED_INT, None)
    else:
      glDrawElementsInstanced(element_type, ne, GL.GL_UNSIGNED_INT, None, ninst)

def glDrawElementsInstanced(mode, count, etype, indices, ninst):
  if bool(GL.glDrawElementsInstanced):
    # OpenGL 3.1 required for this call.
    GL.glDrawElementsInstanced(mode, count, etype, indices, ninst)
  else:
    from OpenGL.GL.ARB.draw_instanced import glDrawElementsInstancedARB
    if not bool(glDrawElementsInstancedARB):
      # Mac 10.6 does not list draw_instanced as an extension using OpenGL 3.2
      from .pyopengl_draw_instanced import glDrawElementsInstancedARB
    glDrawElementsInstancedARB(mode, count, etype, indices, ninst)

def glVertexAttribDivisor(attr_id, d):
    if bool(GL.glVertexAttribDivisor):
      GL.glVertexAttribDivisor(attr_id, d)  # OpenGL 3.3
    else:
      from OpenGL.GL.ARB.instanced_arrays import glVertexAttribDivisorARB
      glVertexAttribDivisorARB(attr_id, d)

def union_bounds(blist):
  xyz_min, xyz_max = None, None
  for b in blist:
    if b is None or b == (None, None):
      continue
    pmin, pmax = b
    if xyz_min is None:
      xyz_min, xyz_max = pmin, pmax
    else:
      xyz_min = tuple(min(x,px) for x,px in zip(xyz_min, pmin))
      xyz_max = tuple(max(x,px) for x,px in zip(xyz_max, pmax))
  return xyz_min, xyz_max

def copies_bounding_box(bounds, plist):
  (x0,y0,z0),(x1,y1,z1) = bounds
  corners = ((x0,y0,z0),(x1,y0,z0),(x0,y1,z0),(x1,y1,z0),
             (x0,y0,z1),(x1,y0,z1),(x0,y1,z1),(x1,y1,z1))
  from . import matrix
  b = union_bounds(point_bounds(matrix.apply_matrix(p, corners)) for p in plist)
  return b

def point_bounds(xyz):
  if len(xyz) == 0:
    return None
  from numpy import array
  axyz = array(xyz)
  return axyz.min(axis = 0), axyz.max(axis = 0)

def surface_image(qi, pos, size, surf = None):
    rgba = image_rgba_array(qi)
    if surf is None:
        surf = Surface()
    p = surf.newPiece()
    x,y = pos
    sx,sy = size
    from numpy import array, float32, uint32
    vlist = array(((x,y,0),(x+sx,y,0),(x+sx,y+sy,0),(x,y+sy,0)), float32)
    tlist = array(((0,1,2),(0,2,3)), uint32)
    tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
    p.geometry = vlist, tlist
    p.useLighting = False
    p.textureCoordinates = tc
    p.textureId = texture_2d(rgba)
    return p

# Extract rgba values from a QImage.
def image_rgba_array(i):
    s = i.size()
    w,h = s.width(), s.height()
    from .qt import QtGui
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

def opengl_matrices(m34_list):
  n = len(m34_list)
  from numpy import empty, float32, transpose
  m = empty((n,4,4), float32)
  m[:,:,:3] = transpose(m34_list, axes = (0,2,1))
  m[:,:3,3] = 0
  m[:,3,3] = 1
  return m

def texture_2d(data):

  from OpenGL import GL
  t = GL.glGenTextures(1)
  GL.glBindTexture(GL.GL_TEXTURE_2D, t)
  h, w = data.shape[:2]
  format, iformat, tdtype = texture_format(data)
  GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
  GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, iformat, w, h, 0,
                  format, tdtype, data)

  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
#  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
#  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

  ncomp = data.shape[2]
  if ncomp == 1 or ncomp == 2:
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
  if ncomp == 2:
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_GREEN)
  GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

  return t

def reload_texture(t, data):

  h, w = data.shape[:2]
  format, iformat, tdtype = texture_format(data)
  from OpenGL import GL
  GL.glBindTexture(GL.GL_TEXTURE_2D, t)
  GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, format, tdtype, data)
  GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

def texture_format(data):

  from OpenGL import GL
  if len(data.shape) == 2 and data.itemsize == 4:
    format = GL.GL_RGBA
    iformat = GL.GL_RGBA8
    tdtype = GL.GL_UNSIGNED_BYTE
    return format, iformat, tdtype
  
  ncomp = data.shape[2]
  # TODO: Report pyopengl bug, GL_RG missing
  GL.GL_RG = 0x8227
  # luminance texture formats are not in opengl 3.
  format = {1:GL.GL_RED, 2:GL.GL_RG,
            3:GL.GL_RGB, 4:GL.GL_RGBA}[ncomp]
  iformat = {1:GL.GL_RED, 2:GL.GL_RG,
             3:GL.GL_RGB8, 4:GL.GL_RGBA8}[ncomp]
  dtype = data.dtype
  from numpy import uint8, float32
  if dtype == uint8:
    tdtype = GL.GL_UNSIGNED_BYTE
  elif dtype == float32:
    tdtype = GL.GL_FLOAT
  else:
    raise TypeError('Texture value type %s not supported' % str(dtype))
  return format, iformat, tdtype
