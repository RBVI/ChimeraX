'''
draw: OpenGL graphics
=====================

All calls to OpenGL are made through this module.  Currently all OpenGL is done with PyOpenGL.

The Render class manages shader, view matrices, and lighting.  The Buffer class handles object
geometry (vertices, normals, triangles) and colors and texture coordinates.  The Bindings class
defines the connections between Buffers and shader program variables.  The Texture class manages 2D
texture storage.
'''

from OpenGL import GL

class Render:
    '''
    Manage shaders, viewing matrices and lighting parameters to render a scene.
    '''
    def __init__(self):
                
        self.shader_programs = {}
        self.current_shader_program = None

        self.default_capabilities = set((self.SHADER_LIGHTING,self.SHADER_VERTEX_COLORS))
        self.override_capabilities = {}

        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        self.current_model_matrix = None        # Used for optimizing model view matrix updates
        self.current_inv_view_matrix = None        # Used for optimizing model view matrix updates

        self.lighting_params = Lighting()

        self.framebuffer_stack = []
        self.mask_framebuffer = None
        self.outline_framebuffer = None

        # Texture warp parameters
        self.warp_center = (0.5, 0.5)
        self.radial_warp_coefficients = (1,0,0,0)
        self.chromatic_warp_coefficients = (1,0,1,0)

        self.single_color = (1,1,1,1)

    # use_shader() option names
    SHADER_LIGHTING = 'USE_LIGHTING'
    SHADER_DEPTH_CUE = 'USE_DEPTH_CUE'
    SHADER_TEXTURE_2D = 'USE_TEXTURE_2D'
    SHADER_RADIAL_WARP = 'USE_RADIAL_WARP'
    SHADER_SHIFT_AND_SCALE = 'USE_INSTANCING_SS'
    SHADER_INSTANCING = 'USE_INSTANCING_44'
    SHADER_TEXTURE_MASK = 'USE_TEXTURE_MASK'
    SHADER_VERTEX_COLORS = 'USE_VERTEX_COLORS'

    def set_override_capabilities(self, ocap):
        self.override_capabilities = ocap

    def use_shader(self, options):
        '''
        Set the shader to use that supports the specified capabilities needed.
        The capabilities are privided as keyword options with boolean values.
        The available option names are given by the values of SHADER_LIGHTING,
        SHADER_TEXTURE_2D, SHADER_SHIFT_AND_SCALE, SHADER_INSTANCING.
        '''

        capabilities = self.default_capabilities.copy()
        ocap = self.override_capabilities
        if ocap:
            options = options.copy()
            options.update(ocap)
        for opt,onoff in options.items():
            if onoff:
                capabilities.add(opt)
            else:
                capabilities.discard(opt)

        p = self.opengl_shader(capabilities)

        if p != self.current_shader_program:
#            print ('changed shader',
#                   self.current_shader_program.capabilities if self.current_shader_program else None, p.capabilities)
            self.current_shader_program = p
            GL.glUseProgram(p.program_id)
            if self.SHADER_LIGHTING in capabilities:
                self.set_shader_lighting_parameters()
            if self.SHADER_DEPTH_CUE in capabilities:
                self.set_depth_cue_parameters()
            self.set_projection_matrix()
            self.set_model_view_matrix()
            if self.SHADER_TEXTURE_2D in capabilities:
                GL.glUniform1i(p.uniform_id("tex2d"), 0)    # Texture unit 0.
            if self.SHADER_RADIAL_WARP in capabilities:
                self.set_radial_warp_parameters()
            if not self.SHADER_VERTEX_COLORS in capabilities:
                self.set_single_color()

        return p

    def push_framebuffer(self, fb):
        self.framebuffer_stack.append(fb)
        fb.activate()
        self.set_drawing_region(0,0,fb.width,fb.height)

    def current_framebuffer(self):
        s = self.framebuffer_stack
        return s[-1] if s else None

    def pop_framebuffer(self):
        s = self.framebuffer_stack
        s.pop()
        if len(s) == 0:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        else:
            fb = s[-1]
            fb.activate()

    def rendering_to_screen(self):
        return len(self.framebuffer_stack) == 0

    def opengl_shader(self, capabilities = (SHADER_LIGHTING,), glsl_version = '150'):
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

    def set_model_view_matrix(self, view_matrix_inverse = None, model_matrix = None):
        '''
        Set the shader to use matrix as the given 4x4 OpenGL model view matrix.
        Or if matrix is not specified use the given model_matrix and view_matrix Place objects
        to calculate the model view matrix.
        '''

        if model_matrix is None:
            mv4 = self.current_model_view_matrix
            if mv4 is None:
                return
        else:
            if model_matrix == self.current_model_matrix:
                from numpy import all
                if all(view_matrix_inverse == self.current_inv_view_matrix):
                    return
            self.current_inv_view_matrix = v = view_matrix_inverse
            self.current_model_matrix = m = model_matrix
            # TODO: optimize matrix multiply.  Rendering bottleneck with 200 models open.
            self.current_model_view_matrix = mv4 = (v*m).opengl_matrix()

        p = self.current_shader_program
        if not p is None:
            var_id = p.uniform_id('model_view_matrix')
            # Note: Getting about 5000 glUniformMatrix4fv() calls per second on 2013 Mac hardware.
            # This can be a rendering bottleneck for large numbers of models or instances.
            GL.glUniformMatrix4fv(var_id, 1, False, mv4)
            if not self.lighting_params.move_lights_with_camera:
                self.set_shader_lighting_parameters()

    def set_shader_lighting_parameters(self):
        'Private. Sets shader lighting variables using the lighting parameters object given in the contructor.'

        p = self.current_shader_program.program_id
        lp = self.lighting_params

        move = None if lp.move_lights_with_camera else self.current_inv_view_matrix

        # Key light
        key_light_dir = GL.glGetUniformLocation(p, b"key_light_direction")
        kld = tuple(move.apply_without_translation(lp.key_light_direction)) if move else lp.key_light_direction
        GL.glUniform3f(key_light_dir, *kld)
        key_diffuse = GL.glGetUniformLocation(p, b"key_light_diffuse_color")
        GL.glUniform3f(key_diffuse, *lp.key_light_diffuse_color)

        # Key light specular
        key_specular = GL.glGetUniformLocation(p, b"key_light_specular_color")
        GL.glUniform3f(key_specular, *lp.key_light_specular_color)
        key_shininess = GL.glGetUniformLocation(p, b"key_light_specular_exponent")
        GL.glUniform1f(key_shininess, lp.key_light_specular_exponent)

        # Fill light
        fill_light_dir = GL.glGetUniformLocation(p, b"fill_light_direction")
        fld = tuple(move.apply_without_translation(lp.fill_light_direction)) if move else lp.fill_light_direction 
        GL.glUniform3f(fill_light_dir, *fld)
        fill_diffuse = GL.glGetUniformLocation(p, b"fill_light_diffuse_color")
        GL.glUniform3f(fill_diffuse, *lp.key_light_diffuse_color)

        # Ambient light
        ambient = GL.glGetUniformLocation(p, b"ambient_color")
        GL.glUniform3f(ambient, *lp.ambient_light_color)

    def set_depth_cue_parameters(self):
        'Private. Sets shader depth variables using the lighting parameters object given in the contructor.'

        p = self.current_shader_program.program_id
        lp = self.lighting_params

        dc_distance = GL.glGetUniformLocation(p, b"depth_cue_distance")
        GL.glUniform1f(dc_distance, lp.depth_cue_distance)
        dc_darkest = GL.glGetUniformLocation(p, b"depth_cue_darkest")
        GL.glUniform1f(dc_darkest, lp.depth_cue_darkest)

    def set_single_color(self, color = None):
        '''
        Set the OpenGL shader color for shader single color mode.
        '''
        if not color is None:
            self.single_color = color
        p = self.current_shader_program.program_id
        c = GL.glGetUniformLocation(p, b"color")
        GL.glUniform4fv(c, 1, self.single_color)

    def set_radial_warp_parameters(self):
        p = self.current_shader_program.program_id
        wcenter = GL.glGetUniformLocation(p, b"warp_center")
        GL.glUniform2fv(wcenter, 1, self.warp_center)
        rcoef = GL.glGetUniformLocation(p, b"radial_warp")
        GL.glUniform4fv(rcoef, 1, self.radial_warp_coefficients)
        ccoef = GL.glGetUniformLocation(p, b"chromatic_warp")
        GL.glUniform4fv(ccoef, 1, self.chromatic_warp_coefficients)

    def opengl_version(self):
        'String description of the OpenGL version for the current context.'
        return GL.glGetString(GL.GL_VERSION).decode('utf-8')

    def support_stereo(self):
        'Return if sequential stereo is supported.'
        return GL.glGetBoolean(GL.GL_STEREO)

    def initialize_opengl(self):
        'Create an initial vertex array object.'

        # OpenGL 3.2 core profile requires a bound vertex array object
        # for drawing, or binding shader attributes to VBOs.  Mac 10.8
        # gives an error if no VAO is bound when glCompileProgram() called.
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

    def set_drawing_region(self, x, y, w, h):
        'Set the OpenGL viewport.'
        GL.glViewport(x, y, w, h)

    def set_background_color(self, rgba):
        'Set the OpenGL clear color.'
        r,g,b,a = rgba
        GL.glClearColor(r, g, b, a)

    def draw_background(self):
        'Draw the background color and clear the depth buffer.'
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def enable_depth_test(self, enable):
        'Enable OpenGL depth testing.'
        if enable:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

    def enable_blending(self, enable):
        'Enable OpenGL alpha blending.'
        if enable:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

    def draw_transparent(self, draw_depth, draw):
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

    def frame_buffer_image(self, w, h, format = IMAGE_FORMAT_RGBA8):
        '''
        Return the current frame buffer image as a numpy array of size (h,w) for 32-bit
        formats or (h,w,4) for 8-bit formats where w and h are the framebuffer width and height.
        Array index 0,0 is at the bottom left corner of the OpenGL viewport for RGB32 format
        and at the upper left corner for the other formats.  For 32-bit formats the array values
        are uint32 and contain 8-bit red, green, and blue values is the low 24 bits for RGB32,
        and 8-bit red, green, blue and alpha for RGBA32.  The RGBA8 format has uint8 values.
        '''

        if format == self.IMAGE_FORMAT_RGBA32:
            from numpy import empty, uint32
            rgba = empty((h,w),uint32)
            GL.glReadPixels(0,0,w,h,GL.GL_RGBA, GL.GL_UNSIGNED_INT_8_8_8_8, rgba)
            return rgba
        elif format == self.IMAGE_FORMAT_RGB32:
            rgba = self.frame_buffer_image(w, h, self.IMAGE_FORMAT_RGBA32)
            rgba >>= 8
            rgb = rgba[::-1,:].copy()
            return rgb
        elif format == self.IMAGE_FORMAT_RGBA8:
            rgba = self.frame_buffer_image(w, h, self.IMAGE_FORMAT_RGBA32)
            from numpy import little_endian, uint8
            if little_endian:
                rgba.byteswap(True) # in place
                rgba8 = rgba.view(uint8).reshape((h,w,4))
                return rgba8

    def set_stereo_buffer(self, eye_num):
        '''Set the draw and read buffers for the left eye (0) or right eye (0).'''
        if not self.rendering_to_screen():
            return
        b = GL.GL_BACK_LEFT if eye_num == 0 else GL.GL_BACK_RIGHT
        GL.glDrawBuffer(b)
        GL.glReadBuffer(b)

    def set_mono_buffer(self):
        '''Set the draw and read buffers for mono rendering.'''
        if not self.rendering_to_screen():
            return
        b = GL.GL_BACK
        GL.glDrawBuffer(b)
        GL.glReadBuffer(b)

    def start_rendering_outline(self, size):

        fb = self.current_framebuffer()
        size = (fb.width, fb.height) if fb else size
        mfb = self.make_mask_framebuffer(size)
        self.push_framebuffer(mfb)
        self.set_background_color((0,0,0,0))
        self.draw_background()
        # Use flat single color rendering.
        self.set_override_capabilities({self.SHADER_VERTEX_COLORS:False,
                                        self.SHADER_LIGHTING:False,
                                        self.SHADER_TEXTURE_2D:False})
        self.set_depth_range(0,0.999)      # Depth test GL_LEQUAL results in z-fighting
        mfb.copy_depth_from_another_framebuffer(fb)

    def finish_rendering_outline(self):

        self.pop_framebuffer()
        self.set_override_capabilities({})
        self.set_depth_range(0,1)
        t = self.mask_framebuffer.texture
        self.draw_texture_mask_outline(t)

    def make_mask_framebuffer(self, size):
        mfb = self.mask_framebuffer
        w,h = size
        if mfb and mfb.width == w and mfb.height == h:
            return mfb
        if mfb:
            mfb.delete()
        t = Texture()
        t.initialize_8_bit(w,h)
        self.mask_framebuffer = mfb = Framebuffer(texture = t)
        return mfb

    def make_outline_framebuffer(self, size):
        ofb = self.outline_framebuffer
        w,h = size
        if ofb and ofb.width == w and ofb.height == h:
            return ofb
        if ofb:
            ofb.delete()
        t = Texture()
        t.initialize_8_bit(w,h)
        self.outline_framebuffer = ofb = Framebuffer(texture = t, depth = False)
        return ofb

    def draw_texture_mask_outline(self, texture, color = (0,1,0,1)):

        # Draw to a new texture 4 shifted copies of texture and erase an unshifted copy to produce an outline.
        ofb = self.make_outline_framebuffer(texture.size)
        self.push_framebuffer(ofb)
        self.set_background_color((0,0,0,0))
        self.draw_background()

        # Render region with texture red > 0.
        self.use_shader({self.SHADER_TEXTURE_MASK:True,
                         self.SHADER_LIGHTING:False,
                         self.SHADER_VERTEX_COLORS:False})

        # Texture map a full-screen quad to blend texture with frame buffer.
        tc = Texture_Copier(self.current_shader_program)
        texture.bind_texture()

        # Draw 4 shifted copies of mask
        w,h = texture.size
        dx, dy = 1.0/w, 1.0/h
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.set_texture_mask_color((1,1,1,1))
        for xs,ys in ((-dx,-dy), (dx,-dy), (dx,dy), (-dx,dy)):
            tc.draw_shifted(xs,ys)

        # Erase unshifted copy of mask
        GL.glBlendFunc(GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        tc.draw_shifted(0,0)

        # Now outline is in texture of the outline framebuffer
        outline = ofb.texture
        self.pop_framebuffer()

        # Draw outline on original framebuffer
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        outline.bind_texture()
        self.set_texture_mask_color(color)
        tc.draw_shifted(0,0)
        
    def set_texture_mask_color(self, color):

        p = self.current_shader_program.program_id
        mc = GL.glGetUniformLocation(p, b"color")
        GL.glUniform4fv(mc, 1, color)

    def set_depth_range(self, min, max):
#        GL.glDepthFunc(GL.GL_LEQUAL)   # Get z-fighting with screen depth copied to framebuffer object on Mac/Nvidia
        GL.glDepthRange(min, max)

    def finish_rendering(self):
        GL.glFinish()

class Framebuffer:

    def __init__(self, width = None, height = None, texture = None, depth = True):

        w,h = (width, height) if texture is None else texture.size
        
        self.width = w
        self.height = h
        self.texture = texture

        c,d,f = self.init_buffer(w,h,depth) if texture is None else self.init_using_texture(texture,depth)

        self.fbo = f
        self.color_rb = c
        self.depth_rb = d

    def __del__(self):

        self.delete()

    def init_buffer(self, width, height, depth):

        max_rb_size = GL.glGetInteger(GL.GL_MAX_RENDERBUFFER_SIZE)
        max_tex_size = GL.glGetInteger(GL.GL_MAX_TEXTURE_SIZE)
        max_size = min(max_rb_size, max_tex_size)
        if width > max_size or height > max_size:
            return None, None, None

        color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, width, height)

        depth_rb, fbo = self.create_fbo(color_rb, width, height, False, depth)
        return color_rb, depth_rb, fbo

    def init_using_texture(self, texture, depth):

        w, h = texture.size
        depth_rb, fbo = self.create_fbo(texture.id, w, h, True, depth)
        return None, depth_rb, fbo

    def create_fbo(self, color_buf, width, height, to_texture, depth):

        # Create color and depth buffers
        if depth:
            depth_rb = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
        else:
            depth_rb = None

        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        if to_texture:
            level = 0
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, color_buf, level)
        else:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                         GL.GL_RENDERBUFFER, color_buf)

        if depth:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                         GL.GL_RENDERBUFFER, depth_rb)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self.delete()
            # TODO: Need to rebind previous framebuffer.
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            return None

        return depth_rb, fbo

    def valid(self):
        return not self.fbo is None

    def activate(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

    def delete(self):
        if self.fbo is None:
            return

        if not self.color_rb is None:
            GL.glDeleteRenderbuffers(1, (self.color_rb,))
        if not self.depth_rb is None:
            GL.glDeleteRenderbuffers(1, (self.depth_rb,))
        GL.glDeleteFramebuffers(1, (self.fbo,))
        self.color_rb = self.depth_rb = self.fbo = None

    def copy_depth_from_another_framebuffer(self, framebuffer):
        # Copy screen depth buffer to fbo
        sfbo = framebuffer.fbo if framebuffer else 0
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, sfbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        w, h = self.width, self.height
        GL.glBlitFramebuffer(0,0,w,h, 0,0,w,h, GL.GL_DEPTH_BUFFER_BIT, GL.GL_NEAREST)

class Lighting:
    '''
    Lighting parameters specifying colors and positions of two lights:
    a key (main) light, and a fill light, as well as specular lighting color
    and exponent and an ambient light color.

      key_light_direction
      key_light_diffuse_color
      key_light_specular_color
      key_light_specular_exponent
      fill_light_direction
      fill_light_diffuse_color
      ambient_light_color

    Directions are unit vectors in camera coordinates (x right, y up, z opposite camera view).
    Colors are R,G,B float values in the range 0-1, and specular exponent is a single float
    value used as an exponent e with specular color scaled by cosine(a) ** e where a is the
    angle between the reflected light and the view direction.  A typical value for e is 30.
    '''

    def __init__(self):
        # Lighting parameters
        self.key_light_direction = (.577,-.577,-.577)    # Should have unit length
        self.key_light_diffuse_color = (.6,.6,.6)
        self.key_light_specular_color = (.3,.3,.3)
        self.key_light_specular_exponent = 30
        self.fill_light_direction = (-.2,-.2,-.959)        # Should have unit length
        self.fill_light_diffuse_color = (.3,.3,.3)
        self.ambient_light_color = (.3,.3,.3)

        self.depth_cue_distance = 15.0  # Distance where dimming begins (Angstroms)
        self.depth_cue_darkest = 0.2    # Smallest dimming factor

        self.move_lights_with_camera = True

class Bindings:
    '''
    Use an OpenGL vertex array object to save buffer bindings.
    The bindings are for a specific shader program since they use the shader variable ids.
    '''
    def __init__(self, shader = None):
        self.shader = shader
        self.vao_id = GL.glGenVertexArrays(1)
        self.bound_attr_ids = {}        # Maps buffer to list of ids

    def __del__(self):
        'Delete the OpenGL vertex array object.'
        GL.glDeleteVertexArrays(1, (self.vao_id,))

    def activate(self):
        'Activate the bindings by binding the OpenGL vertex array object.'
        GL.glBindVertexArray(self.vao_id)

    def bind_shader_variable(self, buffer):
        '''
        Bind the shader variable associated with buffer to the buffer data.
        This enables the OpenGL attribute array, and enables instancing if
        the buffer is instance data.
        '''
        buf_id = buffer.opengl_buffer
        btype = buffer.buffer_type
        if buf_id is None:
            # Unbind already bound variable
            for a in self.bound_attr_ids.get(buffer,[]):
                GL.glDisableVertexAttribArray(a)
            self.bound_attr_ids[buffer] = []
            if buffer.buffer_type == GL.GL_ELEMENT_ARRAY_BUFFER:
                GL.glBindBuffer(btype, 0)
            return

        vname = buffer.shader_variable_name
        if vname is None:
            if btype == GL.GL_ELEMENT_ARRAY_BUFFER:
                # Element array buffer binding is saved in VAO.
                GL.glBindBuffer(btype, buf_id)
            return

        shader = self.shader
        for cap in buffer.requires_capabilities:
            if not cap in shader.capabilities:
                return

        attr_id = shader.attribute_id(vname)
        if attr_id == -1:
            raise NameError('Failed to find shader attribute %s\n in shader with capabilites = %s'
                            % (vname, str(shader.capabilities)))
        nattr = buffer.attribute_count()
        ncomp = buffer.component_count()
        from numpy import float32, uint8
        gtype = {float32:GL.GL_FLOAT,
                 uint8:GL.GL_UNSIGNED_BYTE}[buffer.value_type]
        normalize = GL.GL_TRUE if buffer.normalize else GL.GL_FALSE

        GL.glBindBuffer(btype, buf_id)
        if nattr == 1:
            GL.glVertexAttribPointer(attr_id, ncomp, gtype, normalize, 0, None)
            GL.glEnableVertexAttribArray(attr_id)
            glVertexAttribDivisor(attr_id, 1 if buffer.instance_buffer else 0)
            self.bound_attr_ids[buffer] = [attr_id]
        else:
            # Matrices use multiple vector attributes
            esize = buffer.array_element_bytes()
            abytes = ncomp * esize
            stride = nattr * abytes
            import ctypes
            for a in range(nattr):
                # Pointer arg must be void_p, not an integer.
                p = ctypes.c_void_p(a*abytes)
                GL.glVertexAttribPointer(attr_id+a, ncomp, gtype, normalize, stride, p)
                GL.glEnableVertexAttribArray(attr_id+a)
                glVertexAttribDivisor(attr_id+a, 1 if buffer.instance_buffer else 0)
            self.bound_attr_ids[buffer] = [attr_id+a for a in range(nattr)]
        GL.glBindBuffer(btype, 0)

        return attr_id

from numpy import uint8, uint32, float32

class Buffer_Type:
    def __init__(self, shader_variable_name,
                 buffer_type = GL.GL_ARRAY_BUFFER, value_type = float32,
                 normalize = False, instance_buffer = False, requires_capabilities = ()):
        self.shader_variable_name = shader_variable_name
        self.buffer_type = buffer_type
        self.value_type = value_type
        self.normalize = normalize
        self.instance_buffer = instance_buffer
        self.requires_capabilities = requires_capabilities

# Buffer types with associated shader variable names
VERTEX_BUFFER = Buffer_Type('position')
NORMAL_BUFFER = Buffer_Type('normal', requires_capabilities = (Render.SHADER_LIGHTING,))
VERTEX_COLOR_BUFFER = Buffer_Type('vcolor', value_type = uint8, normalize = True,
                                  requires_capabilities = (Render.SHADER_VERTEX_COLORS,))
INSTANCE_SHIFT_AND_SCALE_BUFFER = Buffer_Type('instanceShiftAndScale', instance_buffer = True)
INSTANCE_MATRIX_BUFFER = Buffer_Type('instancePlacement', instance_buffer = True)
INSTANCE_COLOR_BUFFER = Buffer_Type('vcolor', instance_buffer = True, value_type = uint8, normalize = True,
                                    requires_capabilities = (Render.SHADER_VERTEX_COLORS,))
TEXTURE_COORDS_2D_BUFFER = Buffer_Type('tex_coord_2d', requires_capabilities = (Render.SHADER_TEXTURE_2D,))
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
        self.requires_capabilities = t.requires_capabilities

    def __del__(self):
        self.delete_buffer()

    def delete_buffer(self):
        'Delete the OpenGL buffer object.'

        if self.opengl_buffer is None:
            return
        GL.glDeleteBuffers(1, [self.opengl_buffer])
        self.opengl_buffer = None
        self.buffered_array = None
        self.buffered_data = None

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

    def update_buffer_data(self, data):
        '''
        Update the buffer with data supplied by a numpy array and bind it to the
        associated shader variable.
        '''
        bdata = self.buffered_data
        if data is bdata:
            return False

        replace_buffer = (data is None or bdata is None or data.shape != bdata.shape)
        if replace_buffer:
            self.delete_buffer()

        if not data is None:
            b = GL.glGenBuffers(1) if replace_buffer else self.opengl_buffer
            btype = self.buffer_type
            GL.glBindBuffer(btype, b)
            d = data if data.dtype == self.value_type else data.astype(self.value_type)
            size = d.size * d.itemsize        # Bytes
            if replace_buffer:
                GL.glBufferData(btype, size, d, GL.GL_STATIC_DRAW)
            else:
                # TODO:PyOpenGL-20130502 has glBufferSubData() has python 3 bug, long undefined
                #   So use size None so size is computed from array.
                GL.glBufferSubData(btype, 0, None, d)
            GL.glBindBuffer(btype, 0)
            self.opengl_buffer = b
            self.buffered_array = d
            self.buffered_data = data

        return True

    # Element types for Buffer draw_elements()
    triangles = GL.GL_TRIANGLES
    lines = GL.GL_LINES
    points = GL.GL_POINTS

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
#            print('attrib id for %s is %d, shader %d, cap %s' % (name, aid, p, self.capabilities))
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

        # msg = (('Compiled shader %d,\n'
        #        ' capbilities %s,\n'
        #        ' vertex shader compile info log\n'
        #        ' %s\n'
        #        ' fragment shader compile info log\n'
        #        ' %s\n'
        #        ' program link info log\n'
        #        ' %s')
        #        % (prog_id, capabilities,
        #           GL.glGetShaderInfoLog(vs), GL.glGetShaderInfoLog(fs), GL.glGetProgramInfoLog(prog_id)))
        # print(msg)

        return prog_id

# Add #define lines after #version line of shader
def insert_define_macros(shader, capabilities):
    'Private. Puts "#define" statements in shader program templates to specify shader capabilities.'
    defs = '\n'.join('#define %s 1' % c for c in capabilities)
    v = shader.find('#version')
    eol = shader[v:].find('\n')+1
    s = shader[:eol] + defs + '\n' + shader[eol:]
    return s

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
    def __init__(self, data = None):

        self.id = None

        if not data is None:
            h, w = data.shape[:2]
            format, iformat, tdtype, ncomp = self.texture_format(data)
            self.initialize_texture(w, h, format, iformat, tdtype, ncomp, data)

    def initialize_rgba(self, w, h):

        format = GL.GL_BGRA
        iformat = GL.GL_RGBA8
        tdtype = GL.GL_UNSIGNED_BYTE
        ncomp = 4
        self.initialize_texture(w, h, format, iformat, tdtype, ncomp)

    def initialize_8_bit(self, w, h):

        format = GL.GL_RED
        # TODO: PyOpenGL-20130502 does not have GL_R8.
        GL_R8 = 0x8229
        iformat = GL_R8
        tdtype = GL.GL_UNSIGNED_BYTE
        ncomp = 1
        self.initialize_texture(w, h, format, iformat, tdtype, ncomp)

    def initialize_texture(self, w, h, format, iformat, tdtype, ncomp, data = None):

        from OpenGL import GL
        self.id = t = GL.glGenTextures(1)
        self.size = (w,h)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if data is None:
            import ctypes
            data = ctypes.c_void_p(0)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, iformat, w, h, 0,
                        format, tdtype, data)

        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, (0,0,0,0))
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
#        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

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
        format, iformat, tdtype, ncomp = self.texture_format(data)
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
            ncomp = 4
            return format, iformat, tdtype, ncomp

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
        return format, iformat, tdtype, ncomp

class Texture_Copier:
    '''Draw a texture on a full window rectangle.'''
    def __init__(self, shader):

        self.vao = vao = Bindings(shader)
        vao.activate()
        self.vertex_buf = vb = Buffer(VERTEX_BUFFER)
        from numpy import array, float32, int32
        vb.update_buffer_data(array(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)), float32))
        vao.bind_shader_variable(vb)
        self.tex_coord_buf = tcb = Buffer(TEXTURE_COORDS_2D_BUFFER)
        tcb.update_buffer_data(array(((0,0,0),(1,0,0),(1,1,0),(0,1,0)), float32))
        vao.bind_shader_variable(tcb)
        self.element_buf = eb = Buffer(ELEMENT_BUFFER)
        eb.update_buffer_data(array(((0,1,2),(0,2,3)), int32))
        vao.bind_shader_variable(eb)    # Binds element buffer for rendering

    def __del__(self):
        self.vao = None
        for b in (self.vertex_buf, self.tex_coord_buf, self.element_buf):
            b.delete_buffer()

    def draw_shifted(self, xshift, yshift):
        xs, ys = xshift, yshift
        tcb = self.tex_coord_buf
        from numpy import array, float32
        tcb.update_buffer_data(array(((xs,ys,0),(1+xs,ys,0),(1+xs,1+ys,0),(xs,1+ys,0)), float32))
        eb = self.element_buf
        eb.draw_elements(eb.triangles)
