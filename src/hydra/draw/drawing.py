'''
draw: OpenGL graphics
=====================

All calls to OpenGL are made through this module.  Currently all OpenGL is done with PyOpenGL.
'''

from OpenGL import GL

triangles = GL.GL_TRIANGLES
lines = GL.GL_LINES
points = GL.GL_POINTS

def opengl_version():
    'String description of the OpenGL version for the current context.'
    return GL.glGetString(GL.GL_VERSION).decode('utf-8')

def initialize_opengl():
    'Create an initial vertex array object.'

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
    'Set the OpenGL viewport.'
    GL.glViewport(x, y, w, h)

def set_background_color(rgba):
    'Set the OpenGL clear color.'
    r,g,b,a = rgba
    GL.glClearColor(r, g, b, a)

def draw_background():
    'Draw the background color and clear the depth buffer.'
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

def enable_depth_test(enable):
    'Enable OpenGL depth testing.'
    if enable:
        GL.glEnable(GL.GL_DEPTH_TEST)
    else:
        GL.glDisable(GL.GL_DEPTH_TEST)

def enable_blending(enable):
    'Enable OpenGL alpha blending.'
    if enable:
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    else:
        GL.glDisable(GL.GL_BLEND)

def draw_tile_outlines(tiles, edge_color, fill_color, fill):
    '''
    Draw rectangles with 1 pixel wide borders.  The border is colored
    edge_color and the center is fill_color.  Each rectangle, also called a
    tile is specified by (x,y,w,h) corner and width and height integer pixel
    positions in the OpenGL viewport.  The array of boolean values fill
    says whether a tile should be filled with the fill_color.  If not filled
    it gets the edge color.  The first tile is always filled.  This quirky
    routine is used for tiled display of models.
    '''

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
    '''
    Render using single-layer transparency. This is a two-pass drawing.
    In the first pass is only sets the depth buffer, but not colors, and in
    the second path it draws the colors for pixels at or in front of the
    recorded depths.  The draw_depth and draw routines, taking no arguments
    perform the actual drawing, and are invoked by this routine after setting
    the appropriate OpenGL color and depth drawing modes.
    '''
    # Single layer transparency
    GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
    draw_depth()
    GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    draw()
    GL.glDepthFunc(GL.GL_LESS)

IMAGE_FORMAT_RGBA32 = 'rgba32'
IMAGE_FORMAT_RGBA8 = 'rgba8'
IMAGE_FORMAT_RGB32 = 'rgb32'

def frame_buffer_image(w, h, format = IMAGE_FORMAT_RGBA8):
    '''
    Return the current frame buffer image as a numpy array of size (h,w) for 32-bit
    formats or (h,w,4) for 8-bit formats where w and h are the framebuffer width and height.
    Array index 0,0 is at the bottom left corner of the OpenGL viewport for RGB32 format
    and at the upper left corner for the other formats.  For 32-bit formats the array values
    are uint32 and contain 8-bit red, green, and blue values is the low 24 bits for RGB32,
    and 8-bit red, green, blue and alpha for RGBA32.  The RGBA8 format has uint8 values.
    '''

    if format == IMAGE_FORMAT_RGBA32:
        from numpy import empty, uint32
        rgba = empty((h,w),uint32)
        GL.glReadPixels(0,0,w,h,GL.GL_RGBA, GL.GL_UNSIGNED_INT_8_8_8_8, rgba)
        return rgba
    elif format == IMAGE_FORMAT_RGB32:
        rgba = frame_buffer_image(w, h, IMAGE_FORMAT_RGBA32)
        rgba >>= 8
        rgb = rgba[::-1,:].copy()
        return rgb
    elif format == IMAGE_FORMAT_RGBA8:
        rgba = frame_buffer_image(w, h, IMAGE_FORMAT_RGBA32)
        from numpy import little_endian, uint8
        if little_endian:
            rgba.byteswap(True) # in place
        rgba8 = rgba.view(uint8).reshape((h,w,4))
        return rgba8

class Texture:
    '''
    Create an OpenGL 2d texture from a numpy array of of size (h,w,c) or (h,w)
    where w and h are the texture width and height and c is the number of color components.
    If the data array is 2-dimensional, the values must be 32-bit RGBA8.  If the data
    array is 3 dimensional the texture format is GL_RED, GL_RG, GL_RGB, or GL_RGBA depending
    on whether c is 1, 2, 3 or 4 and only value types of uint8 or float32 are allowed and
    texture of type GL_UNSIGNED_BYTE or GL_FLOAT is created.  Clamp to edge mode and
    nearest interpolation is set.  The c = 2 mode uses the second component as alpha and
    the first componet for red,green,blue.
    '''
    def __init__(self, data):

        from OpenGL import GL
        self.id = t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        h, w = data.shape[:2]
        format, iformat, tdtype = self.texture_format(data)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, iformat, w, h, 0,
                        format, tdtype, data)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        ncomp = data.shape[2]
        if ncomp == 1 or ncomp == 2:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
        if ncomp == 2:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_GREEN)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):
        'Delete the OpenGL texture.'
        GL.glDeleteTextures((self.id,))

    def bind_texture(self):
        'Bind the OpenGL 2d texture.'
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)

    def unbind_texture(self):
        'Unbind the OpenGL 2d texture.'
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def reload_texture(self, data):
        '''
        Replace the texture values in texture with OpenGL id using numpy array data.
        The data is interpreted the same as for the texture_2d() function.
        '''

        h, w = data.shape[:2]
        format, iformat, tdtype = self.texture_format(data)
        from OpenGL import GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, format, tdtype, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def texture_format(self, data):
        '''
        Return the OpenGL texture format, internal format, and texture value type
        that will be used by the texture_2d() function when creating a texture from
        a numpy array of colors.
        '''
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

class Bindings:
    '''
    Use an OpenGL vertex array object to save buffer bindings.
    '''
    def __init__(self):
        self.vao_id = GL.glGenVertexArrays(1)
    def __del__(self):
        'Delete the OpenGL vertex array object.'
        GL.glDeleteVertexArrays(1, (self.vao_id,))
    def bind(self):
        'Bind the OpenGL vertex array object.'
        GL.glBindVertexArray(self.vao_id)

def set_single_color(shader_prog, color):
    '''
    Set the OpenGL shader color for single color drawing to the specified
    r,g,b,a float values.
    '''
    r,g,b,a = color
    GL.glVertexAttrib4f(shader_prog.attribute_id("vcolor"), r,g,b,a)

from numpy import uint8, uint32, float32

class Buffer_Type:
    def __init__(self, shader_variable_name,
                 buffer_type = GL.GL_ARRAY_BUFFER, value_type = float32,
                 normalize = False, instance_buffer = False):
        self.shader_variable_name = shader_variable_name
        self.buffer_type = buffer_type
        self.value_type = value_type
        self.normalize = normalize
        self.instance_buffer = instance_buffer

# Buffer types with associated shader variable names
VERTEX_BUFFER = Buffer_Type('position')
NORMAL_BUFFER = Buffer_Type('normal')
INSTANCE_SHIFT_AND_SCALE_BUFFER = Buffer_Type('instanceShiftAndScale', instance_buffer = True)
INSTANCE_MATRIX_BUFFER = Buffer_Type('instancePlacement', instance_buffer = True)
VERTEX_COLOR_BUFFER = Buffer_Type('vcolor', value_type = uint8, normalize = True)
INSTANCE_COLOR_BUFFER = Buffer_Type('vcolor', instance_buffer = True, value_type = uint8, normalize = True)
TEXTURE_COORDS_2D_BUFFER = Buffer_Type('tex_coord_2d')
ELEMENT_BUFFER = Buffer_Type(None, buffer_type = GL.GL_ELEMENT_ARRAY_BUFFER, value_type = uint32)

class Buffer:
    '''
    Create an OpenGL buffer of vertex data such as vertex positions, normals, or colors,
    or per-instance data (e.g. color per sphere) or an element buffer for specifying which
    primitives (triangles, lines, points) to draw.  Vertex data buffers can be attached to
    a specific shader variable.
    '''
    def __init__(self, buffer_type):

        t = buffer_type
        self.shader_variable_name = t.shader_variable_name
        self.opengl_buffer = None
        self.buffered_array = None  # numpy array for vbo
        self.buffered_data = None   # data need not be numpy array
        self.value_type = t.value_type
        self.buffer_type = t.buffer_type
        self.normalize = t.normalize
        self.instance_buffer = t.instance_buffer
        self.bound_attr_ids = []

    def update_buffer_data(self, data):
        'Private: Load the buffer with the specified data from a numpy array.'

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
        'Delete the OpenGL buffer object.'

        if self.opengl_buffer is None:
            return
        GL.glDeleteBuffers(1, [self.opengl_buffer])
        self.opengl_buffer = None
        self.buffered_array = None
        self.buffered_data = None

    def bind_shader_variable(self, shader):
        'Private.'
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
        'Private.'
        # matrix attributes use multiple attribute ids
        barray = self.buffered_array
        if barray is None:
            return 0
        bshape = barray.shape
        nattr = 1 if len(bshape) == 2 else bshape[1]
        return nattr

    def component_count(self):
        'Private.'
        barray = self.buffered_array
        if barray is None:
            return 0
        ncomp = barray.shape[-1]
        return ncomp
  
    def array_element_bytes(self):
        'Private.'
        barray = self.buffered_array
        return 0 if barray is None else barray.itemsize

    def update_buffer(self, data, shader, new_vao):
        '''
        Update the buffer with data supplied by a numpy array.
        Use the specified shader object for binding the buffer to a shader variable.
        If new_vao is true, then assume the vertex array object saving the shader
        variable bindings has changed and update the shader variable binding.
        '''
        if new_vao:
            self.bound_attr_ids = []

        if self.update_buffer_data(data) or new_vao:
            self.bind_shader_variable(shader)

    def draw_elements(self, element_type = GL.GL_TRIANGLES, ninst = None):
        '''
        Draw primitives using this buffer as the element buffer.
        All the required buffers are assumed to be already bound using a
        vertex array object.
        '''
        # Don't bind element buffer since it is bound by VAO.
        ne = self.buffered_array.size
        if ninst is None:
            GL.glDrawElements(element_type, ne, GL.GL_UNSIGNED_INT, None)
        else:
            glDrawElementsInstanced(element_type, ne, GL.GL_UNSIGNED_INT, None, ninst)

def glDrawElementsInstanced(mode, count, etype, indices, ninst):
    'Private. Handle old or defective OpenGL instanced drawing.'
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
    'Private. Handle old or defective OpenGL attribute divisor.'
    if bool(GL.glVertexAttribDivisor):
        GL.glVertexAttribDivisor(attr_id, d)  # OpenGL 3.3
    else:
        from OpenGL.GL.ARB.instanced_arrays import glVertexAttribDivisorARB
        glVertexAttribDivisorARB(attr_id, d)

# Render.use_shader() boolean options
SHADER_LIGHTING = 'lighting'
SHADER_TEXTURE_2D = 'texture2d'
SHADER_SHIFT_AND_SCALE = 'shiftAndScale'
SHADER_INSTANCING = 'instancing'
SHADER_SELECTED = 'selected'
SHADER_UNSELECTED = 'unselected'

class Render:
    '''
    Manage shaders, viewing matrices and lighting parameters to render a scene.

    Lighting parameters is an object specifying colors and positions of two
    lights: a key (main) light, and a fill light, as well as specular lighting
    color and exponent and an ambient light color.  These are attributes of the
    specified lighting_params object named

      key_light_position
      key_light_diffuse_color
      key_light_specular_color
      key_light_specular_exponent
      fill_light_position
      fill_light_diffuse_color
      ambient_light_color

    Colors are R,G,B float values in the range 0-1, positions are x,y,z float values,
    and specular exponent is a single float value used as an exponent e with specular
    color scaled by cosine(a) ** 0.3*e where a is the angle between the reflected light
    and the view direction.  A typical value for e is 20.
    '''
    def __init__(self, lighting_params):
                
        self.shader_programs = {}
        self.current_shader_program = None

        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        self.current_model_matrix = None        # Used for optimizing model view matrix updates
        self.current_inv_view_matrix = None        # Used for optimizing model view matrix updates

        self.lighting_params = lighting_params

    def use_shader(self, **kw):
        '''
        Set the shader to use that supports the specified capabilities needed.
        The capabilities are privided as keyword options with boolean values.
        The available option names are given by the values of SHADER_LIGHTING,
        SHADER_TEXTURE_2D, SHADER_SHIFT_AND_SCALE, SHADER_INSTANCING,
        SHADER_SELECTED, SHADER_UNSELECTED.
        '''

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
        'Private.  OpenGL shader program id.'

        ckey = tuple(sorted(capabilities))
        p = self.shader_programs.get(ckey)
        if not p is None:
            return p

        p = Shader(capabilities, glsl_version)
        self.shader_programs[ckey] = p

        return p
        
    def set_projection_matrix(self, pm = None):
        '''
        Set the shader to use the given 4x4 OpenGL projection matrix.
        If no matrix is specified use the last specified one.
        '''
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
        '''
        Set the shader to use matrix as the given 4x4 OpenGL model view matrix.
        Or if matrix is not specified use the given model_matrix and view_matrix Place objects
        to calculate the model view matrix.
        '''

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
        'Private. Sets shader lighting variables using the lighting parameters object given in the contructor.'

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
    '''Private. OpenGL shader program with specified capabilities.'''

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
    'Private. Puts "#define" statements in shader program templates to specify shader capabilities.'
    defs = '\n'.join('#define %s 1' % c for c in capabilities)
    v = shader.find('#version')
    eol = shader[v:].find('\n')+1
    s = shader[:eol] + defs + '\n' + shader[eol:]
    return s
