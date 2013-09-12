from OpenGL import GL

triangles = GL.GL_TRIANGLES
lines = GL.GL_LINES
points = GL.GL_POINTS
element_array = GL.GL_ELEMENT_ARRAY_BUFFER

def opengl_version():

    return GL.glGetString(GL.GL_VERSION).decode('utf-8')

def initialize_opengl():

    # OpenGL 3.2 core profile requires a bound vertex array object
    # for drawing, or binding shader attributes to VBOs.  Mac 10.8
    # gives an error if no VAO is bound when glCompileProgram() called.
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)

def depth_buffer_size():

# TODO: GL_DEPTH_BITS is deprecated or removed in OpenGL 3
#    return GL.glGetInteger(GL.GL_DEPTH_BITS)

    # TODO: never got this to work.  Got invalid operation indicating that
    # no render buffer is bound.  Got this even in the paintGL() routine.
    import ctypes
    s = ctypes.c_int()
    GL.glGetRenderbufferParameteriv(GL.GL_RENDERBUFFER,
                                    GL.GL_RENDERBUFFER_DEPTH_SIZE,
                                    ctypes.byref(s))
    return s.value()

def set_drawing_region(x, y, w, h):

    GL.glViewport(x, y, w, h)

def set_background_color(rgba):

    r,g,b,a = rgba
    GL.glClearColor(r, g, b, a)

def draw_background():

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

def enable_depth_test(enable):

    if enable:
        GL.glEnable(GL.GL_DEPTH_TEST)
    else:
        GL.glDisable(GL.GL_DEPTH_TEST)

def enable_blending(enable):

    if enable:
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    else:
        GL.glDisable(GL.GL_BLEND)

def draw_tile_outlines(tiles, edge_color, fill_color, fill):

    GL.glClearColor(*edge_color)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glClearColor(*fill_color)
    n = len(fill)
    for i, (tx,ty,tw,th) in enumerate(tiles):
        if i == 0 or i > n or fill[i-1]:
            if i == 0 or i > n:
                GL.glScissor(tx,ty,tw,th)
            else:
                GL.glScissor(tx+1,ty+1,tw-2,th-2)
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glDisable(GL.GL_SCISSOR_TEST)

def draw_transparent(draw_depth, draw):

    # Single layer transparency
    GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
    draw_depth()
    GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    draw()
    GL.glDepthFunc(GL.GL_LESS)

def frame_buffer_image_rgb32(w, h):

    rgba = frame_buffer_image_rgba32(w, h)
    rgba >>= 8
    rgb = rgba[::-1,:].copy()
    return rgb

def frame_buffer_image_rgba32(w, h):

    from numpy import empty, uint32
    rgba = empty((h,w),uint32)
    GL.glReadPixels(0,0,w,h,GL.GL_RGBA, GL.GL_UNSIGNED_INT_8_8_8_8, rgba)
    return rgba

def frame_buffer_image_rgba8(w, h):

    rgba = frame_buffer_image_rgba32(w, h)
    from numpy import little_endian, uint8
    if little_endian:
        rgba.byteswap(True) # in place
    rgba8 = rgba.view(uint8).reshape((h,w,4))
    return rgba8

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
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
#  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
#  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

  ncomp = data.shape[2]
  if ncomp == 1 or ncomp == 2:
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
  if ncomp == 2:
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_GREEN)
  GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

  return t

def bind_2d_texture(id):
    GL.glBindTexture(GL.GL_TEXTURE_2D, id)

def delete_texture(id):
    GL.glDeleteTextures((id,))

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

def new_vertex_array_object():
    return GL.glGenVertexArrays(1)
def bind_vertex_array_object(vao):
    GL.glBindVertexArray(vao)
def delete_vertex_array_object(vao):
    GL.glDeleteVertexArrays(1, (vao,))

def set_single_color(shader_prog, color):
    r,g,b,a = color
    GL.glVertexAttrib4f(shader_prog.attribute_id("vcolor"), r,g,b,a)

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

# Renderer.use_shader() boolean options
SHADER_LIGHTING = 'lighting'
SHADER_TEXTURE_2D = 'texture2d'
SHADER_SHIFT_AND_SCALE = 'shiftAndScale'
SHADER_INSTANCING = 'instancing'
SHADER_SELECTED = 'selected'
SHADER_UNSELECTED = 'unselected'

class Renderer:

    def __init__(self, lighting_params):
                
        self.shader_programs = {}
        self.current_shader_program = None

        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        self.current_model_matrix = None        # Used for optimizing model view matrix updates
        self.current_inv_view_matrix = None        # Used for optimizing model view matrix updates

        self.lighting_params = lighting_params

    def use_shader(self, **kw):

        default_capabilities = ('USE_LIGHTING',)
        capabilities = set(default_capabilities)

        options = {SHADER_LIGHTING:'USE_LIGHTING',
                   SHADER_TEXTURE_2D:'USE_TEXTURE_2D',
                   SHADER_SHIFT_AND_SCALE:'USE_INSTANCING_SS',
                   SHADER_INSTANCING:'USE_INSTANCING_44',
                   SHADER_SELECTED:'USE_HATCHING'}
#                   SHADER_UNSELECTED:'USE_DIMMING'}
        for opt,onoff in kw.items():
            if opt in options:
                cap = options[opt]
                if onoff:
                    capabilities.add(cap)
                else:
                    capabilities.discard(cap)

        p = self.opengl_shader(capabilities)

        if p != self.current_shader_program:
#            print ('changed shader',
#                   self.current_shader_program.capabilities if self.current_shader_program else None, p.capabilities)
            self.current_shader_program = p
            GL.glUseProgram(p.program_id)
            if 'USE_LIGHTING' in capabilities:
                self.set_shader_lighting_parameters()
            self.set_projection_matrix()
            self.set_model_view_matrix()
            if 'USE_TEXTURE_2D' in capabilities:
                GL.glUniform1i(p.uniform_id("tex2d"), 0)    # Texture unit 0.

        return p

    def opengl_shader(self, capabilities = ('USE_LIGHTING',), glsl_version = '150'):
        
        ckey = tuple(sorted(capabilities))
        p = self.shader_programs.get(ckey)
        if not p is None:
            return p

        p = Shader(capabilities, glsl_version)
        self.shader_programs[ckey] = p

        return p
        
    def set_projection_matrix(self, pm = None):

        if pm is None:
            if self.current_projection_matrix is None:
                return
            pm = self.current_projection_matrix
        else:
            self.current_projection_matrix = pm
        p = self.current_shader_program
        if not p is None:
            GL.glUniformMatrix4fv(p.uniform_id('projection_matrix'), 1, False, pm)

    def set_model_view_matrix(self, view_matrix_inverse = None, model_matrix = None, matrix = None):

        if not matrix is None:
            mv4 = matrix
            self.current_model_view_matrix = mv4
            self.current_model_matrix = None
            self.current_inv_view_matrix = None
        elif model_matrix is None:
            mv4 = self.current_model_view_matrix
            if mv4 is None:
                return
        else:
            if model_matrix == self.current_model_matrix:
                from numpy import all
                if all(view_matrix_inverse == self.current_inv_view_matrix):
                    return
            v = view_matrix_inverse
            m = model_matrix
            # TODO: optimize matrix multiply.  Rendering bottleneck with 200 models open.
            mv4 = (v*m).opengl_matrix()
            self.current_model_view_matrix = mv4
            self.current_model_matrix = m
            self.current_inv_view_matrix = v

        p = self.current_shader_program
        if not p is None:
            var_id = p.uniform_id('model_view_matrix')
            # Note: Getting about 5000 glUniformMatrix4fv() calls per second on 2013 Mac hardware.
            # This can be a rendering bottleneck for large numbers of models or instances.
            GL.glUniformMatrix4fv(var_id, 1, False, mv4)

    def set_shader_lighting_parameters(self):

        p = self.current_shader_program.program_id
        lp = self.lighting_params

        # Key light
        key_light_pos = GL.glGetUniformLocation(p, b"key_light_position")
        GL.glUniform3f(key_light_pos, *lp.key_light_position)
        key_diffuse = GL.glGetUniformLocation(p, b"key_light_diffuse_color")
        GL.glUniform3f(key_diffuse, *lp.key_light_diffuse_color)

        # Key light specular
        key_specular = GL.glGetUniformLocation(p, b"key_light_specular_color")
        GL.glUniform3f(key_specular, *lp.key_light_specular_color)
        key_shininess = GL.glGetUniformLocation(p, b"key_light_specular_exponent")
        GL.glUniform1f(key_shininess, lp.key_light_specular_exponent)

        # Fill light
        fill_light_pos = GL.glGetUniformLocation(p, b"fill_light_position")
        GL.glUniform3f(fill_light_pos, *lp.fill_light_position)
        fill_diffuse = GL.glGetUniformLocation(p, b"fill_light_diffuse_color")
        GL.glUniform3f(fill_diffuse, *lp.key_light_diffuse_color)

        # Ambient light
        ambient = GL.glGetUniformLocation(p, b"ambient_color")
        GL.glUniform3f(ambient, *lp.ambient_light_color)

class Shader:

    def __init__(self, capabilities, glsl_version = '150'):

        self.capabilities = capabilities
        self.program_id = self.compile_shader(capabilities, glsl_version)
        self.uniform_ids = {}
        self.attribute_ids = {}
        
    def uniform_id(self, name):
        uids = self.uniform_ids
        if name in uids:
            uid = uids[name]
        else:
            p = self.program_id
            uids[name] = uid = GL.glGetUniformLocation(p, name.encode('utf-8'))
        return uid

    def attribute_id(self, name):
        aids = self.attribute_ids
        if name in aids:
            aid = aids[name]
        else:
            p = self.program_id
            aids[name] = aid = GL.glGetAttribLocation(p, name.encode('utf-8'))
        return aid

    def compile_shader(self, capabilities, glsl_version = '150'):

        from os.path import dirname, join
        d = dirname(__file__)
        f = open(join(d,'vshader%s.txt' % glsl_version), 'r')
        vshader = insert_define_macros(f.read(), capabilities)
        f.close()

        f = open(join(d,'fshader%s.txt' % glsl_version), 'r')
        fshader = insert_define_macros(f.read(), capabilities)
        f.close()

        from OpenGL.GL import shaders
        vs = shaders.compileShader(vshader, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fshader, GL.GL_FRAGMENT_SHADER)

        prog_id = shaders.compileProgram(vs, fs)
        return prog_id


# Add #define lines after #version line of shader
def insert_define_macros(shader, capabilities):
    defs = '\n'.join('#define %s 1' % c for c in capabilities)
    v = shader.find('#version')
    eol = shader[v:].find('\n')+1
    s = shader[:eol] + defs + '\n' + shader[eol:]
    return s
