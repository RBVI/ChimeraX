# vi: set expandtab shiftwidth=4 softtabstop=4:
'''
OpenGL classes
==============

All calls to OpenGL are made through this module.  Currently all OpenGL
is done with PyOpenGL.

The Render class manages shader, view matrices, and lighting.  The Buffer
class handles object geometry (vertices, normals, triangles) and colors
and texture coordinates.  The Bindings class defines the connections
between Buffers and shader program variables.  The Texture class manages
2D texture storage.  '''

def configure_offscreen_rendering():
    from chimera import core, app_lib_dir
    if not hasattr(core, 'offscreen_rendering'):
        return
    import sys
    if sys.platform == 'darwin':
        return  # OSMesa 10.6.3 with LLVM crashes in glDrawElements() on Mac.
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    # OSMesa 10.6.2 gives an OpenGL 3.0 compatibility context on Linux with only GLSL 130 support.
    # Chimera needs OpenGL 3.3 with GLSL 330 shaders and requires only a core context.
    # Overriding Mesa to give OpenGL 3.3, gives a core context that really does support 3.3.
    # TODO: This would not be necessary if Linux Chimera requested a core context.
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    # Tell PyOpenGL where to find libOSMesa
    lib_suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    os.environ['PYOPENGL_OSMESA_LIB_PATH'] = os.path.join(app_lib_dir, 'libOSMesa' + lib_suffix)

# Set environment variables set before importing PyOpenGL.
configure_offscreen_rendering()

from OpenGL import GL

# OpenGL workarounds:
stencil8_needed = False

class Render:
    '''
    Manage shaders, viewing matrices and lighting parameters to render a scene.
    '''
    def __init__(self):

        self.shader_programs = {}
        self.current_shader_program = None

        self.enable_capabilities = 0    # Bit field of SHADER_* capabilities
        self.disable_capabilities = 0

        self.current_viewport = None
        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        # Used for optimizing model view matrix updates:
        self.current_model_matrix = None
        # Maps scene to camera coordinates:
        self.current_view_matrix = None

        self.lighting = Lighting()
        self.material = Material()              # Currently a global material

        self._default_framebuffer = None
        self.framebuffer_stack = [self.default_framebuffer()]
        self.mask_framebuffer = None
        self.outline_framebuffer = None
        self._silhouette_framebuffer = None

        # 3D ambient texture transform from model coordinates to texture
        # coordinates:
        self.ambient_texture_transform = None

        # Shadows
        self.shadow_map_framebuffer = None
        self.shadow_texture_unit = 1
        # Maps camera coordinates to shadow map texture coordinates:
        self._shadow_transform = None
        self.multishadow_map_framebuffer = None
        self.multishadow_texture_unit = 2
        self._max_multishadows = None
        self._multishadow_transforms = None
        # near to far clip depth for shadow map:
        self._multishadow_depth = None
        # Uniform buffer object for shadow matrices:
        self._multishadow_matrix_buffer = None
        self._multishadow_uniform_block = 0     # Uniform block number

        self.single_color = (1, 1, 1, 1)
        self.frame_number = 0

    def default_framebuffer(self):
        if self._default_framebuffer is None:
            self._default_framebuffer = Framebuffer(color=False, depth=False)
        return self._default_framebuffer

    def render_size(self):
        fb = self.current_framebuffer()
        x, y, w, h = fb.viewport
        return (w, h)

    def disable_shader_capabilities(self, ocap):
        self.disable_capabilities = ocap

    def draw_depth_only(self, depth_only=True):
        # Enable only shader geometry, no colors or lighting.
        if depth_only:
            d = ~(self.SHADER_INSTANCING | self.SHADER_SHIFT_AND_SCALE)
        else:
            d = 0
        self.disable_capabilities = d
        c = GL.GL_FALSE if depth_only else GL.GL_TRUE
        GL.glColorMask(c, c, c, c)

    def shader(self, options):
        '''
        Return a shader that supports the specified capabilities.
        The capabilities are specified as at bit field of values from
        SHADER_LIGHTING, SHADER_DEPTH_CUE, SHADER_TEXTURE_2D,
        SHADER_TEXTURE_3D_AMBIENT, SHADER_SHADOWS, SHADER_MULTISHADOW,
        SHADER_SHIFT_AND_SCALE, SHADER_INSTANCING, SHADER_TEXTURE_MASK,
        SHADER_DEPTH_OUTLINE, SHADER_VERTEX_COLORS
        '''
        options |= self.enable_capabilities
        options &= ~self.disable_capabilities
        p = self.opengl_shader(options)
        return p

    def use_shader(self, shader):
        '''
        Set the current shader.
        '''
        if shader == self.current_shader_program:
            return

        # print('changed shader', ', '.join(o for i, o in
        #                                   enumerate(shader_options)
        #                                   if shader.capabilities & (1 << i)))
        self.current_shader_program = shader
        c = shader.capabilities
        GL.glUseProgram(shader.program_id)
        if self.SHADER_LIGHTING & c:
            self.set_shader_lighting_parameters()
            if self.SHADER_TEXTURE_3D_AMBIENT & c:
                shader.set_integer('tex3d', 0)    # Tex unit 0.
            if self.SHADER_MULTISHADOW & c:
                self.set_shadow_shader_variables(shader)
            if self.SHADER_SHADOWS & c:
                shader.set_integer("shadow_map", self.shadow_texture_unit)
                if self._shadow_transform is not None:
                    shader.set_matrix("shadow_transform", self._shadow_transform)
            if self.SHADER_DEPTH_CUE & c:
                self.set_depth_cue_parameters()
        self.set_projection_matrix()
        self.set_model_matrix()
        if (self.SHADER_TEXTURE_2D & c or self.SHADER_TEXTURE_MASK & c
                or self.SHADER_DEPTH_OUTLINE & c):
            shader.set_integer("tex2d", 0)    # Texture unit 0.
        if not self.SHADER_VERTEX_COLORS & c:
            self.set_single_color()
        if self.SHADER_FRAME_NUMBER & c:
            self.set_frame_number()

    def push_framebuffer(self, fb):
        self.framebuffer_stack.append(fb)
        fb.activate()
        self.set_viewport(*fb.viewport)

    def current_framebuffer(self):
        return self.framebuffer_stack[-1]

    def pop_framebuffer(self):
        s = self.framebuffer_stack
        pfb = s.pop()
        if len(s) == 0:
            raise RuntimeError('No framebuffer left on stack.')
        fb = s[-1]
        fb.activate()
        self.set_viewport(*fb.viewport)
        return pfb

    def rendering_to_screen(self):
        return len(self.framebuffer_stack) == 1

    def opengl_shader(self, capabilities):
        'Private.  OpenGL shader program id.'

        sp = self.shader_programs
        if capabilities in sp:
            p = sp[capabilities]
        else:
            p = Shader(capabilities, self.max_multishadows())
            sp[capabilities] = p

        return p

    def set_projection_matrix(self, pm=None):
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
        if p is not None:
            p.set_matrix('projection_matrix', pm)

    def set_view_matrix(self, vm):
        '''Set the camera view matrix, mapping scene to camera coordinates.'''
        self.current_view_matrix = vm
        self.current_model_matrix = None
        self.current_model_view_matrix = None

    def set_model_matrix(self, model_matrix=None):
        '''
        Set the shader model view using the given model matrix and
        previously set view matrix.  If no matrix is specified, the
        shader gets the last used one model view matrix.
        '''

        if model_matrix is None:
            mv4 = self.current_model_view_matrix
            if mv4 is None:
                return
        else:
            # TODO: optimize check of same model matrix.
            cmm = self.current_model_matrix
            if cmm:
                if ((model_matrix.is_identity() and cmm.is_identity())
                        or model_matrix.same(cmm)):
                    return
            self.current_model_matrix = model_matrix
            # TODO: optimize matrix multiply.  Rendering bottleneck with 200
            # models open.
            mv4 = (self.current_view_matrix * model_matrix).opengl_matrix()
            self.current_model_view_matrix = mv4

        p = self.current_shader_program
        if p is not None:
            p.set_matrix('model_view_matrix', mv4)
#            if p.capabilities & self.SHADER_MULTISHADOW:
#                m4 = self.current_model_matrix.opengl_matrix()
#                p.set_matrix('model_matrix', m4)
            if not self.lighting.move_lights_with_camera:
                self.set_shader_lighting_parameters()

    def set_frame_number(self, f=None):
        if f is None:
            f = self.frame_number
        else:
            self.frame_number = f
        p = self.current_shader_program
        if p is not None and self.SHADER_FRAME_NUMBER & p.capabilities:
            p.set_float('frame_number', f)

    def set_shader_lighting_parameters(self):
        '''Private. Sets shader lighting variables using the lighting
        parameters object given in the contructor.'''

        p = self.current_shader_program
        if p is None:
            return

        lp = self.lighting
        mp = self.material

        move = None if lp.move_lights_with_camera else self.current_view_matrix

        # Key light
        kld = move.apply_without_translation(lp.key_light_direction) if move else lp.key_light_direction
        p.set_vector("key_light_direction", kld)
        ds = mp.diffuse_reflectivity * lp.key_light_intensity
        kdc = tuple(ds * c for c in lp.key_light_color)
        p.set_vector("key_light_diffuse_color", kdc)

        # Key light specular
        ss = mp.specular_reflectivity * lp.key_light_intensity
        ksc = tuple(ss * c for c in lp.key_light_color)
        p.set_vector("key_light_specular_color", ksc)
        p.set_float("key_light_specular_exponent", mp.specular_exponent)

        # Fill light
        fld = move.apply_without_translation(lp.fill_light_direction) if move else lp.fill_light_direction
        p.set_vector("fill_light_direction", fld)
        ds = mp.diffuse_reflectivity * lp.fill_light_intensity
        fdc = tuple(ds * c for c in lp.fill_light_color)
        p.set_vector("fill_light_diffuse_color", fdc)

        # Ambient light
        ams = mp.ambient_reflectivity * lp.ambient_light_intensity
        ac = tuple(ams * c for c in lp.ambient_light_color)
        p.set_vector("ambient_color", ac)

    def set_depth_cue_parameters(self):
        '''Private. Sets shader depth variables using the lighting
        parameters object given in the contructor.'''

        p = self.current_shader_program
        if p is None:
            return

        lp = self.lighting
        p.set_float("depth_cue_distance", lp.depth_cue_distance)
        p.set_float("depth_cue_darkest", lp.depth_cue_darkest)

    def set_single_color(self, color=None):
        '''
        Set the OpenGL shader color for shader single color mode.
        '''
        if color is not None:
            self.single_color = color
        p = self.current_shader_program
        if p is not None:
            p.set_rgba("color", self.single_color)

    def set_ambient_texture_transform(self, tf):
        # Transform from model coordinates to ambient texture coordinates.
        p = self.current_shader_program
        if p is not None:
            p.set_matrix("ambient_tex3d_transform", tf.opengl_matrix())

    def set_shadow_transform(self, stf):
        # Transform from camera coordinates to shadow map texture coordinates.
        self._shadow_transform = m = stf.opengl_matrix()
        p = self.current_shader_program
        if p is not None and self.SHADER_SHADOWS & p.capabilities:
            p.set_matrix("shadow_transform", m)

    def set_multishadow_transforms(self, stf, ctf, shadow_depth):
        # Transform from camera coordinates to shadow map texture coordinates.
        from numpy import array, float32
        self._multishadow_transforms = array([(tf * ctf).opengl_matrix()
                                              for tf in stf], float32)
#        self._multishadow_transforms = array([tf.opengl_matrix()
#                                                  for tf in stf], float32)
        self._multishadow_depth = shadow_depth
        p = self.current_shader_program
        if p is not None:
            c = p.capabilities
            if self.SHADER_MULTISHADOW & c and self.SHADER_LIGHTING & c:
                self.set_shadow_shader_variables(p)

    def set_shadow_shader_variables(self, shader):
        shader.set_integer("multishadow_map", self.multishadow_texture_unit)
        m = self._multishadow_transforms
        if m is None:
            return

        # Setup uniform buffer object for shadow matrices.
        # It can have larger size than an array of uniforms.
        b = self._multishadow_matrix_buffer
        maxs = self.max_multishadows()
        if b is None:
            self._multishadow_matrix_buffer = b = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, b)
            GL.glBufferData(GL.GL_UNIFORM_BUFFER, maxs * 64,
                            pyopengl_null(), GL.GL_DYNAMIC_DRAW)
            bi = GL.glGetUniformBlockIndex(shader.program_id, b'shadow_matrix_block')
            GL.glUniformBlockBinding(shader.program_id, bi, self._multishadow_uniform_block)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, self._multishadow_uniform_block, b)
        # TODO: Issue warning if maximum number of shadows exceeded.
        mm = m[:maxs, :, :]
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 0, len(mm) * 64, mm)
#        shader.set_matrices("shadow_transforms", m)
        shader.set_integer("shadow_count", len(mm))
        shader.set_float("shadow_depth", self._multishadow_depth)

    def max_multishadows(self):
        'Maximum number of shadows to cast.'
        m = self._max_multishadows
        if m is None:
            m = GL.glGetIntegerv(GL.GL_MAX_UNIFORM_BLOCK_SIZE)      # OpenGL requires >= 16384.
            m = m // 64                                             # 64 bytes per matrix.
            self._max_multishadows = m
        return m

    def opengl_version(self):
        'String description of the OpenGL version for the current context.'
        return GL.glGetString(GL.GL_VERSION).decode('utf-8')

    def opengl_info(self):
        lines = ['vendor: %s' % GL.glGetString(GL.GL_VENDOR).decode('utf-8'),
                 'renderer: %s' % GL.glGetString(GL.GL_RENDERER).decode('utf-8'),
                 'version: %s' % GL.glGetString(GL.GL_VERSION).decode('utf-8'),
                 'GLSL version: %s' % GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode('utf-8')]
        ne = GL.glGetInteger(GL.GL_NUM_EXTENSIONS)
        for e in range(ne):
            lines.append('extension: %s' % GL.glGetStringi(GL.GL_EXTENSIONS,e).decode('utf-8'))
        return '\n'.join(lines)

    def support_stereo(self):
        'Return if sequential stereo is supported.'
        return GL.glGetBoolean(GL.GL_STEREO)

    def opengl_context_changed(self):
        'Called after opengl context is switched.'
        p = self.current_shader_program
        if p is not None:
            GL.glUseProgram(p.program_id)

    def initialize_opengl(self, width, height):
        'Create an initial vertex array object.'

        # OpenGL 3.2 core profile requires a bound vertex array object
        # for drawing, or binding shader attributes to VBOs.  Mac 10.8
        # gives an error if no VAO is bound when glCompileProgram() called.
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        fb = self.default_framebuffer()
        fb.width, fb.height = width, height
        self.set_viewport(0, 0, width, height)

        # Detect OpenGL workarounds
        vendor = GL.glGetString(GL.GL_VENDOR)
        import sys
        global stencil8_needed
        stencil8_needed = (sys.platform.startswith('linux') and
                           vendor.startswith((b'AMD', b'ATI')))

    def set_viewport(self, x, y, w, h):
        'Set the OpenGL viewport.'
        if (x, y, w, h) != self.current_viewport:
            GL.glViewport(x, y, w, h)
            self.current_viewport = (x, y, w, h)
        fb = self.current_framebuffer()
        fb.viewport = (x, y, w, h)

    def full_viewport(self):
        fb = self.current_framebuffer()
        self.set_viewport(0, 0, fb.width, fb.height)

    def set_background_color(self, rgba):
        'Set the OpenGL clear color.'
        r, g, b, a = rgba
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

    def blend_add(self, f):
        GL.glBlendColor(f, f, f, f)
        GL.glBlendFunc(GL.GL_CONSTANT_COLOR, GL.GL_ONE)
        GL.glEnable(GL.GL_BLEND)

    def draw_transparent(self, draw_depth, draw):
        '''
        Render using single-layer transparency. This is a two-pass
        drawing.  In the first pass is only sets the depth buffer,
        but not colors, and in the second path it draws the colors for
        pixels at or in front of the recorded depths.  The draw_depth and
        draw routines, taking no arguments perform the actual drawing,
        and are invoked by this routine after setting the appropriate
        OpenGL color and depth drawing modes.
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

    def frame_buffer_image(self, w, h, rgba = None, front_buffer = False):
        '''
        Return the current frame buffer image as a numpy uint8 array of
        size (h, w, 4) where w and h are the framebuffer width and height.
        The four components are red, green, blue, alpha.  Array index 0,
        0 is at the bottom left corner of the OpenGL viewport.
        '''
        if rgba is None:
            from numpy import empty, uint8
            rgba = empty((h, w, 4), uint8)
        if front_buffer:
            GL.glReadBuffer(GL.GL_FRONT)
        GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
        if front_buffer:
            GL.glReadBuffer(GL.GL_BACK)
        return rgba

    def set_stereo_buffer(self, eye_num):
        '''Set the draw and read buffers for the left eye (0) or right
        eye (1).'''
        self.full_viewport()
        if not self.rendering_to_screen():
            return
        b = GL.GL_BACK_LEFT if eye_num == 0 else GL.GL_BACK_RIGHT
        GL.glDrawBuffer(b)
        GL.glReadBuffer(b)

    def set_mono_buffer(self):
        '''Set the draw and read buffers for mono rendering.'''
        self.full_viewport()
        if not self.rendering_to_screen():
            return
        b = GL.GL_BACK
        GL.glDrawBuffer(b)
        GL.glReadBuffer(b)

    def start_rendering_shadowmap(self, center, radius, size=1024):

        fb = self.shadowmap_start(self.shadow_map_framebuffer,
                                  self.shadow_texture_unit, center, radius,
                                  size)
        self.shadow_map_framebuffer = fb

    def start_rendering_multishadowmap(self, center, radius, size=1024):

        fb = self.shadowmap_start(self.multishadow_map_framebuffer,
                                  self.multishadow_texture_unit, center,
                                  radius, size)
        self.multishadow_map_framebuffer = fb

    def shadowmap_start(self, framebuffer, texture_unit, center, radius, size):

        # Set projection matrix to be orthographic and span all models.
        from ..geometry import scale, translation
        pm = (scale((1 / radius, 1 / radius, -1 / radius))
              * translation((0, 0, radius)))  # orthographic projection along z
        self.set_projection_matrix(pm.opengl_matrix())

        # Make a framebuffer for depth texture rendering
        fb = framebuffer
        if fb is None or fb.width != size:
            if fb:
                fb.delete()
            dt = Texture()
            dt.initialize_depth((size, size))
            fb = Framebuffer(depth_texture=dt, color=False)
            if not fb.valid():
                return           # Requested size exceeds framebuffer limits

        # Make sure depth texture is not bound from previous drawing so that
        # it is not used for rendering shadows while the depth texture is
        # being written.
        # TODO: The depth rendering should not render colors or shadows.
        dt = fb.depth_texture
        dt.unbind_texture(texture_unit)

        # Draw the models recording depth in light direction, i.e., calculate
        # the shadow map.
        self.push_framebuffer(fb)

        self.draw_depth_only()

        return fb

    def finish_rendering_shadowmap(self):

        self.draw_depth_only(False)
        fb = self.pop_framebuffer()
        return fb.depth_texture

    def finish_rendering_multishadowmap(self):

        return self.finish_rendering_shadowmap()

    def shadow_transforms(self, light_direction, center, radius,
                          depth_bias=0.005):

        # Projection matrix, orthographic along z
        from ..geometry import translation, scale, orthonormal_frame
        pm = (scale((1 / radius, 1 / radius, -1 / radius))
              * translation((0, 0, radius)))

        # Compute the view matrix looking along the light direction.
        ld = light_direction
        # Light view frame:
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse()  # Scene to light view coordinates

        # Convert (-1, 1) normalized device coords to (0, 1) texture coords.
        ntf = translation((0.5, 0.5, 0.5 - depth_bias)) * scale(0.5)
        stf = ntf * pm * lvinv               # Scene to shadowmap coordinates

        fb = self.current_framebuffer()
        w, h = fb.width, fb.height
        if self.current_viewport != (0, 0, w, h):
            # Using a subregion of shadow map to handle multiple shadows in
            # one texture.  Map scene coordinates to subtexture.
            x, y, vw, vh = self.current_viewport
            stf = (translation((x / w, y / w, 0)) * scale((vw / w, vh / h, 1))
                   * stf)

        return lvinv, stf

    def start_rendering_outline(self):

        fb = self.current_framebuffer()
        mfb = self.make_mask_framebuffer()
        self.push_framebuffer(mfb)
        self.set_background_color((0, 0, 0, 0))
        self.draw_background()
        # Use flat single color rendering.
        self.disable_shader_capabilities(self.SHADER_VERTEX_COLORS
                                         | self.SHADER_TEXTURE_2D
                                         | self.SHADER_LIGHTING)
        # Depth test GL_LEQUAL results in z-fighting:
        self.set_depth_range(0, 0.999999)
        # Copy depth to outline framebuffer:
        self.copy_from_framebuffer(fb, color=False)

    def finish_rendering_outline(self):

        self.pop_framebuffer()
        self.disable_shader_capabilities(0)
        self.set_depth_range(0, 1)
        t = self.mask_framebuffer.color_texture
        self.draw_texture_mask_outline(t)

    def make_mask_framebuffer(self):
        size = self.render_size()
        mfb = self.mask_framebuffer
        w, h = size
        if mfb and mfb.width == w and mfb.height == h:
            return mfb
        if mfb:
            mfb.delete()
        t = Texture()
        t.initialize_8_bit(size)
        self.mask_framebuffer = mfb = Framebuffer(color_texture=t)
        return mfb

    def make_outline_framebuffer(self, size):
        ofb = self.outline_framebuffer
        w, h = size
        if ofb and ofb.width == w and ofb.height == h:
            return ofb
        if ofb:
            ofb.delete()
        t = Texture()
        t.initialize_8_bit(size)
        self.outline_framebuffer = ofb = Framebuffer(color_texture=t,
                                                     depth=False)
        return ofb

    def draw_texture_mask_outline(self, texture, color=(0, 1, 0, 1)):

        # Draw to a new texture 4 shifted copies of texture and erase an
        # unshifted copy to produce an outline.
        ofb = self.make_outline_framebuffer(texture.size)
        self.push_framebuffer(ofb)
        self.set_background_color((0, 0, 0, 0))
        self.draw_background()

        # Render region with texture red > 0.
        # Texture map a full-screen quad to blend texture with frame buffer.
        tc = TextureWindow(self, self.SHADER_TEXTURE_MASK)
        texture.bind_texture()

        # Draw 4 shifted copies of mask
        w, h = texture.size
        dx, dy = 1.0 / w, 1.0 / h
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.set_texture_mask_color((1, 1, 1, 1))
        for xs, ys in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)):
            tc.draw(xshift=xs, yshift=ys)

        # Erase unshifted copy of mask
        GL.glBlendFunc(GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        tc.draw()

        # Now outline is in texture of the outline framebuffer
        outline = ofb.color_texture
        self.pop_framebuffer()

        # Draw outline on original framebuffer
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        outline.bind_texture()
        self.set_texture_mask_color(color)
        tc.draw()

    def set_texture_mask_color(self, color):

        p = self.current_shader_program
        if p is not None:
            p.set_rgba("color", color)

    def allow_equal_depth(self, equal):
        GL.glDepthFunc(GL.GL_LEQUAL if equal else GL.GL_LESS)

    def set_depth_range(self, min, max):
        # # Get z-fighting with screen depth copied to framebuffer object
        # # on Mac/Nvidia
        # GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthRange(min, max)

    def start_silhouette_drawing(self):
        fb = self.silhouette_framebuffer(self.render_size())
        self.push_framebuffer(fb)

    def finish_silhouette_drawing(self, thickness, color, depth_jump,
                                  perspective_near_far_ratio):
        fb = self.pop_framebuffer()
        self.copy_from_framebuffer(fb, depth=False)
        self.draw_depth_outline(fb.depth_texture, thickness, color, depth_jump,
                                perspective_near_far_ratio)

    def silhouette_framebuffer(self, size=None):
        sfb = self._silhouette_framebuffer
        if size is None:
            return sfb
        if sfb and (size[0] != sfb.width or size[1] != sfb.height):
            sfb.delete()
            sfb = None
        if sfb is None:
            dt = Texture()
            dt.initialize_depth(size, depth_compare_mode=False)
            self._silhouette_framebuffer = sfb = Framebuffer(depth_texture=dt)
        return sfb

    def draw_depth_outline(self, depth_texture, thickness=1,
                           color=(0, 0, 0, 1), depth_jump=0.01,
                           perspective_near_far_ratio=1):

        # Render pixels with depth less than neighbor pixel by at least
        # depth_jump.  Texture map a full-screen quad to blend depth jump
        # pixels with frame buffer.
        tc = TextureWindow(self, self.SHADER_DEPTH_OUTLINE)
        depth_texture.bind_texture()

        # Draw 4 shifted copies of mask
        w, h = depth_texture.size
        dx, dy = 1.0 / w, 1.0 / h
        self.enable_depth_test(False)
        self.enable_blending(True)
        GL.glDepthMask(False)   # Disable depth write
        self.set_depth_outline_color(color)
        for xs, ys in disk_grid(thickness):
            self.set_depth_outline_shift_and_jump(xs * dx, ys * dy, depth_jump,
                                                  perspective_near_far_ratio)
            tc.draw()
        GL.glDepthMask(True)
        self.enable_blending(False)
        self.enable_depth_test(True)

    def set_depth_outline_color(self, color):

        p = self.current_shader_program
        if p is not None:
            p.set_rgba("color", color)

    def set_depth_outline_shift_and_jump(self, xs, ys, depth_jump,
                                         perspective_near_far_ratio):

        p = self.current_shader_program
        if p is not None:
            v = (xs, ys, depth_jump, perspective_near_far_ratio)
            p.set_float4("depth_shift_and_jump", v)

    def copy_from_framebuffer(self, framebuffer, color=True, depth=True):
        # Copy current framebuffer contents to another framebuffer.  This
        # leaves read and draw framebuffers set to the current framebuffer.
        cfb = self.current_framebuffer()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, framebuffer.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, cfb.fbo)
        what = GL.GL_COLOR_BUFFER_BIT if color else 0
        if depth:
            what |= GL.GL_DEPTH_BUFFER_BIT
        w, h = framebuffer.width, framebuffer.height
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, what, GL.GL_NEAREST)
        # Restore read buffer
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, cfb.fbo)

    def finish_rendering(self):
        GL.glFinish()


def disk_grid(radius, exclude_origin=True):
    r = int(radius)
    r2 = radius * radius
    ij = []
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if (i * i + j * j <= r2
                    and (not exclude_origin or i != 0 or j != 0)):
                ij.append((i, j))
    return ij

# Options used with Render.shader()
shader_options = (
    'SHADER_LIGHTING',
    'SHADER_DEPTH_CUE',
    'SHADER_TEXTURE_2D',
    'SHADER_TEXTURE_3D_AMBIENT',
    'SHADER_SHADOWS',
    'SHADER_MULTISHADOW',
    'SHADER_SHIFT_AND_SCALE',
    'SHADER_INSTANCING',
    'SHADER_TEXTURE_MASK',
    'SHADER_DEPTH_OUTLINE',
    'SHADER_VERTEX_COLORS',
    'SHADER_FRAME_NUMBER',
)
for i, sopt in enumerate(shader_options):
    setattr(Render, sopt, 1 << i)


class Framebuffer:
    '''
    OpenGL framebuffer for off-screen rendering.  Allows rendering colors
    and/or depth to a texture.
    '''
    def __init__(self, width=None, height=None,
                 color=True, color_texture=None,
                 depth=True, depth_texture=None):

        if width is not None and height is not None:
            w, h = width, height
        elif color_texture is not None:
            w, h = color_texture.size
        elif depth_texture is not None:
            w, h = depth_texture.size
        else:
            w = h = None

        self.width = w
        self.height = h
        self.viewport = None if w is None else (0, 0, w, h)

        self.color_texture = color_texture
        if not color or color_texture or w is None:
            self.color_rb = None
        else:
            self.color_rb = self.color_renderbuffer(w, h)
        self.depth_texture = depth_texture
        if not depth or depth_texture or w is None:
            self.depth_rb = None
        else:
            self.depth_rb = self.depth_renderbuffer(w, h)

        if w is None:
            fbo = 0
        elif not self.valid_size(w, h):
            fbo = None
        else:
            fbo = self.create_fbo(color_texture or self.color_rb,
                                  depth_texture or self.depth_rb)
        self.fbo = fbo

    def __del__(self):

        self.delete()

    def valid_size(self, width, height):

        max_rb_size = GL.glGetInteger(GL.GL_MAX_RENDERBUFFER_SIZE)
        max_tex_size = GL.glGetInteger(GL.GL_MAX_TEXTURE_SIZE)
        max_size = min(max_rb_size, max_tex_size)
        return width <= max_size and height <= max_size

    def color_renderbuffer(self, width, height):

        color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, width, height)
        return color_rb

    def depth_renderbuffer(self, width, height):

        depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
        if stencil8_needed:
            # AMD driver requires GL_DEPTH24_STENCIL8 for blitting instead of
            # GL_DEPTH_COMPONENT24 even though we don't have any stencil planes
            iformat = GL.GL_DEPTH24_STENCIL8
        else:
            iformat = GL.GL_DEPTH_COMPONENT24
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, iformat, width, height)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
        return depth_rb

    def create_fbo(self, color_buf, depth_buf):

        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        if isinstance(color_buf, Texture):
            level = 0
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                      GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, color_buf.id, level)
        elif color_buf is not None:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                         GL.GL_COLOR_ATTACHMENT0,
                                         GL.GL_RENDERBUFFER, color_buf)

        if isinstance(depth_buf, Texture):
            level = 0
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                      GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D,
                                      depth_buf.id, level)
        elif depth_buf is not None:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                         GL.GL_DEPTH_ATTACHMENT,
                                         GL.GL_RENDERBUFFER, depth_buf)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self.delete()
            # TODO: Need to rebind previous framebuffer.
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            return None

        return fbo

    def valid(self):
        return self.fbo is not None

    def activate(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

    def delete(self):
        if self.fbo is None:
            return

        if self.color_rb is not None:
            GL.glDeleteRenderbuffers(1, (self.color_rb,))
        if self.depth_rb is not None:
            GL.glDeleteRenderbuffers(1, (self.depth_rb,))
        GL.glDeleteFramebuffers(1, (self.fbo,))
        self.color_rb = self.depth_rb = self.fbo = None


class Lighting:
    '''
    Lighting parameters specifying colors and directions of two lights:
    a key (main) light, and a fill light, as well as ambient light color.
    Directions are unit vectors in camera coordinates (x right, y up, z
    opposite camera view).
    Colors are R, G, B float values in the range 0-1.

    :ivar key_light_direction: (.577, -.577, -.577)
    :ivar key_light_color: (1, 1, 1)
    :ivar key_light_intensity: 1
    :ivar fill_light_direction: (-.2, -.2, -.959)
    :ivar fill_light_color: (1, 1, 1)
    :ivar fill_light_intensity: 0.5
    :ivar ambient_light_color: (1, 1, 1)
    :ivar ambient_light_intensity: 0.4
    :ivar move_lights_with_camera: True
    '''

    def __init__(self):

        self.set_default_parameters()

    def set_default_parameters(self):
        '''
        Reset the lighting parameters to default values.
        '''
        from numpy import array, float32
        # Should have unit length:
        self.key_light_direction = array((.577, -.577, -.577), float32)
        '''Direction key light shines in.'''

        self.key_light_color = (1, 1, 1)
        '''Key light color.'''

        self.key_light_intensity = 1
        '''Key light brightness.'''

        # Should have unit length:
        self.fill_light_direction = array((-.2, -.2, -.959), float32)
        '''Direction fill light shines in.'''

        self.fill_light_color = (1, 1, 1)
        '''Fill light color.'''

        self.fill_light_intensity = 0.5
        '''Fill light brightness.'''

        self.ambient_light_color = (1, 1, 1)
        '''Ambient light color.'''

        self.ambient_light_intensity = 0.4
        '''Ambient light brightness.'''

        # Distance where dimming begins (Angstroms):
        self.depth_cue_distance = 15.0
        self.depth_cue_darkest = 0.2    # Smallest dimming factor

        self.move_lights_with_camera = True
        '''Whether lights are attached to camera, or fixed in the scene.'''


class Material:
    '''
    Surface properties that control the reflection of light.
    '''
    def __init__(self):

        self.set_default_parameters()

    def set_default_parameters(self):
        '''
        Reset the material parameters to default values.
        '''

        self.ambient_reflectivity = 0.8
        '''Fraction of ambient light reflected.  Ambient light comes
        from all directions and the amount reflected does not depend on
        the surface orientation of view direction.'''

        self.diffuse_reflectivity = 0.8
        '''Fraction of direction light reflected diffusely, that is
        depending on light angle to surface but not viewing direction.'''

        self.specular_reflectivity = 0.3
        '''Fraction of directional key light reflected specularly,
        that is depending how close reflected light direction is to the
        viewing direction.'''

        self.specular_exponent = 30
        '''Controls the spread of specular light. The specular exponent
        is a single float value used as an exponent e with reflected
        intensity scaled by cosine(a) ** e where a is the angle between
        the reflected light and the view direction. A typical value for
        e is 30.'''


class Bindings:
    '''
    Use an OpenGL vertex array object to save buffer bindings.
    The bindings are for a specific shader program since they use the
    shader variable ids.
    '''
    attribute_id = {'position': 0, 'tex_coord_2d': 1, 'normal': 2, 'vcolor': 3,
                    'instance_shift_and_scale': 4, 'instance_placement': 5}

    def __init__(self):
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
            for a in self.bound_attr_ids.get(buffer, []):
                GL.glDisableVertexAttribArray(a)
            self.bound_attr_ids[buffer] = []
            if btype == GL.GL_ELEMENT_ARRAY_BUFFER:
                GL.glBindBuffer(btype, 0)
            return

        vname = buffer.shader_variable_name
        if vname is None:
            if btype == GL.GL_ELEMENT_ARRAY_BUFFER:
                # Element array buffer binding is saved in VAO.
                GL.glBindBuffer(btype, buf_id)
            return

        attr_id = self.attribute_id[vname]
        nattr = buffer.attribute_count()
        ncomp = buffer.component_count()
        from numpy import float32, uint8
        gtype = {float32: GL.GL_FLOAT,
                 uint8: GL.GL_UNSIGNED_BYTE}[buffer.value_type]
        normalize = GL.GL_TRUE if buffer.normalize else GL.GL_FALSE

        GL.glBindBuffer(btype, buf_id)
        if nattr == 1:
            GL.glVertexAttribPointer(attr_id, ncomp, gtype, normalize, 0, None)
            GL.glEnableVertexAttribArray(attr_id)
            GL.glVertexAttribDivisor(attr_id, 1 if buffer.instance_buffer else 0)
            self.bound_attr_ids[buffer] = [attr_id]
        else:
            # Matrices use multiple vector attributes
            esize = buffer.array_element_bytes()
            abytes = ncomp * esize
            stride = nattr * abytes
            import ctypes
            for a in range(nattr):
                # Pointer arg must be void_p, not an integer.
                p = ctypes.c_void_p(a * abytes)
                GL.glVertexAttribPointer(attr_id + a, ncomp, gtype, normalize,
                                         stride, p)
                GL.glEnableVertexAttribArray(attr_id + a)
                GL.glVertexAttribDivisor(attr_id + a,
                                         1 if buffer.instance_buffer else 0)
            self.bound_attr_ids[buffer] = [attr_id + a for a in range(nattr)]
        GL.glBindBuffer(btype, 0)

        # print('bound shader variable', vname, attr_id, nattr, ncomp)
        return attr_id


def deactivate_bindings():
    GL.glBindVertexArray(0)

from numpy import uint8, uint32, float32


class BufferType:
    '''
    Describes a shader variable and the vertex buffer object value type
    required and what rendering capabilities are required to use this
    shader variable.
    '''
    def __init__(self, shader_variable_name, buffer_type=GL.GL_ARRAY_BUFFER,
                 value_type=float32, normalize=False, instance_buffer=False,
                 requires_capabilities=()):
        self.shader_variable_name = shader_variable_name
        self.buffer_type = buffer_type
        self.value_type = value_type
        self.normalize = normalize
        self.instance_buffer = instance_buffer

        # Requires at least one of these:
        self.requires_capabilities = requires_capabilities

# Buffer types with associated shader variable names
VERTEX_BUFFER = BufferType('position')
NORMAL_BUFFER = BufferType(
    'normal', requires_capabilities=Render.SHADER_LIGHTING)
VERTEX_COLOR_BUFFER = BufferType(
    'vcolor', value_type=uint8, normalize=True,
    requires_capabilities=Render.SHADER_VERTEX_COLORS)
INSTANCE_SHIFT_AND_SCALE_BUFFER = BufferType(
    'instance_shift_and_scale', instance_buffer=True)
INSTANCE_MATRIX_BUFFER = BufferType(
    'instance_placement', instance_buffer=True)
INSTANCE_COLOR_BUFFER = BufferType(
    'vcolor', instance_buffer=True, value_type=uint8, normalize=True,
    requires_capabilities=Render.SHADER_VERTEX_COLORS)
TEXTURE_COORDS_2D_BUFFER = BufferType(
    'tex_coord_2d',
    requires_capabilities=Render.SHADER_TEXTURE_2D | Render.SHADER_TEXTURE_MASK
    | Render.SHADER_DEPTH_OUTLINE)
ELEMENT_BUFFER = BufferType(None, buffer_type=GL.GL_ELEMENT_ARRAY_BUFFER,
                            value_type=uint32)


class Buffer:
    '''
    Create an OpenGL buffer of vertex data such as vertex positions,
    normals, or colors, or per-instance data (e.g. color per sphere)
    or an element buffer for specifying which primitives (triangles,
    lines, points) to draw.  Vertex data buffers can be attached to a
    specific shader variable.
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
        Update the buffer with data supplied by a numpy array and bind it to
        the associated shader variable.  Return true if the buffer is deleted and replaced.
        '''
        bdata = self.buffered_data
        replace_buffer = (data is None or bdata is None
                          or data.shape != bdata.shape)
        if replace_buffer:
            self.delete_buffer()

        if data is not None:
            b = GL.glGenBuffers(1) if replace_buffer else self.opengl_buffer
            btype = self.buffer_type
            GL.glBindBuffer(btype, b)
            if data.dtype == self.value_type:
                d = data
            else:
                d = data.astype(self.value_type)
            size = d.size * d.itemsize        # Bytes
            if replace_buffer:
                GL.glBufferData(btype, size, d, GL.GL_STATIC_DRAW)
            else:
                # TODO: PyOpenGL-20130502 has glBufferSubData() has python 3
                #       bug, long undefined.
                #   So use size None so size is computed from array.
                GL.glBufferSubData(btype, 0, None, d)
            GL.glBindBuffer(btype, 0)
            self.opengl_buffer = b
            self.buffered_array = d
            self.buffered_data = data

        return replace_buffer

    # Element types for Buffer draw_elements()
    triangles = GL.GL_TRIANGLES
    lines = GL.GL_LINES
    points = GL.GL_POINTS

    def draw_elements(self, element_type=triangles, ninst=None):
        '''
        Draw primitives using this buffer as the element buffer.
        All the required buffers are assumed to be already bound using a
        vertex array object.
        '''
        # Don't bind element buffer since it is bound by VAO.
        # TODO: Need to bind it because change to element buffer by update_buffer_data()
        # erases the current binding and I don't have reliable code to restore that binding.
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.opengl_buffer)
        ne = self.buffered_array.size
        if ninst is None:
            GL.glDrawElements(element_type, ne, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawElementsInstanced(element_type, ne, GL.GL_UNSIGNED_INT, None, ninst)

    def shader_has_required_capabilities(self, shader):
        if not self.requires_capabilities:
            return True
        # Require at least one capability of list
        return self.requires_capabilities | shader.capabilities


class Shader:
    '''OpenGL shader program with specified capabilities.'''

    def __init__(self, capabilities, max_shadows):

        self.capabilities = capabilities
        self.program_id = self.compile_shader(capabilities, max_shadows)
        self.uniform_ids = {}

    def set_integer(self, name, value):
        GL.glUniform1i(self.uniform_id(name), value)

    def set_float(self, name, value):
        GL.glUniform1f(self.uniform_id(name), value)

    def set_vector(self, name, vector):
        GL.glUniform3f(self.uniform_id(name), *tuple(vector))

    def set_rgba(self, name, color):
        GL.glUniform4fv(self.uniform_id(name), 1, color)

    def set_float4(self, name, v4):
        GL.glUniform4fv(self.uniform_id(name), 1, v4)

    def set_matrix(self, name, matrix):
        GL.glUniformMatrix4fv(self.uniform_id(name), 1, False, matrix)

    def uniform_id(self, name):
        uids = self.uniform_ids
        if name in uids:
            uid = uids[name]
        else:
            p = self.program_id
            uids[name] = uid = GL.glGetUniformLocation(p, name.encode('utf-8'))
        return uid

    def compile_shader(self, capabilities, max_shadows):

        from os.path import dirname, join
        d = dirname(__file__)
        f = open(join(d, 'vertexShader.txt'), 'r')
        vshader = self.insert_define_macros(f.read(), capabilities, max_shadows)
        f.close()

        f = open(join(d, 'fragmentShader.txt'), 'r')
        fshader = self.insert_define_macros(f.read(), capabilities, max_shadows)
        f.close()

        from OpenGL.GL import shaders
        vs = shaders.compileShader(vshader, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fshader, GL.GL_FRAGMENT_SHADER)

        prog_id = shaders.compileProgram(vs, fs)

        # msg = (('Compiled shader %d,\n'
        #        ' capbilities %s,\n'
        #        ' vertex shader code\n'
        #        ' %s\n'
        #        ' vertex shader compile info log\n'
        #        ' %s\n'
        #        ' fragment shader code\n'
        #        ' %s\n'
        #        ' fragment shader compile info log\n'
        #        ' %s\n'
        #        ' program link info log\n'
        #        ' %s')
        #        % (prog_id, capabilities, vshader, GL.glGetShaderInfoLog(vs),
        #           fshader, GL.glGetShaderInfoLog(fs),
        #           GL.glGetProgramInfoLog(prog_id)))
        # print(msg)

        return prog_id

    # Add #define lines after #version line of shader
    def insert_define_macros(self, shader, capabilities, max_shadows):
        '''Private. Puts "#define" statements in shader program templates
        to specify shader capabilities.'''
        deflines = ['#define %s 1' % sopt.replace('SHADER_', 'USE_')
                    for i, sopt in enumerate(shader_options)
                    if capabilities & (1 << i)]
        deflines.append('#define MAX_SHADOWS %d' % max_shadows)
        defs = '\n'.join(deflines)
        v = shader.find('#version')
        eol = shader[v:].find('\n') + 1
        s = shader[:eol] + defs + '\n' + shader[eol:]
        return s


class Texture:
    '''
    Create an OpenGL 1d, 2d, or 3d texture from a numpy array.  For a
    N dimensional texture the data array can be N or N+1 dimensional.
    For example, for 2d shape (h, w, c) or (h, w) where w and h are the
    texture width and height and c is the number of color components.
    If the data array is 2-dimensional, the values must be 32-bit RGBA8.
    If the data array is 3 dimensional the texture format is GL_RED,
    GL_RG, GL_RGB, or GL_RGBA depending on whether c is 1, 2, 3 or 4
    and only value types of uint8 or float32 are allowed and texture of
    type GL_UNSIGNED_BYTE or GL_FLOAT is created.  Clamp to edge mode
    and nearest interpolation is set.  The c = 2 mode uses the second
    component as alpha and the first componet for red, green, blue.
    '''
    def __init__(self, data=None, dimension=2):

        self.id = None
        self.dimension = dimension
        self.gl_target = (GL.GL_TEXTURE_1D, GL.GL_TEXTURE_2D,
                          GL.GL_TEXTURE_3D)[dimension - 1]

        if data is not None:
            size = tuple(data.shape[dimension - 1::-1])
            format, iformat, tdtype, ncomp = self.texture_format(data)
            self.initialize_texture(size, format, iformat, tdtype, ncomp, data)

    def initialize_rgba(self, size):

        format = GL.GL_BGRA
        iformat = GL.GL_RGBA8
        tdtype = GL.GL_UNSIGNED_BYTE
        ncomp = 4
        self.initialize_texture(size, format, iformat, tdtype, ncomp)

    def initialize_8_bit(self, size):

        format = GL.GL_RED
        # TODO: PyOpenGL-20130502 does not have GL_R8.
        GL_R8 = 0x8229  # noqa
        iformat = GL_R8
        tdtype = GL.GL_UNSIGNED_BYTE
        ncomp = 1
        self.initialize_texture(size, format, iformat, tdtype, ncomp)

    def initialize_depth(self, size, depth_compare_mode=True):

        format = GL.GL_DEPTH_COMPONENT
        if stencil8_needed:
            # for compatibility with glRenderbufferStorage
            iformat = GL.GL_DEPTH24_STENCIL8
        else:
            iformat = GL.GL_DEPTH_COMPONENT24
        tdtype = GL.GL_FLOAT
        ncomp = 1
        self.initialize_texture(size, format, iformat, tdtype, ncomp,
                                depth_compare_mode=depth_compare_mode)

    def initialize_texture(self, size, format, iformat, tdtype, ncomp,
                           data=None, depth_compare_mode=False):

        self.id = t = GL.glGenTextures(1)
        self.size = size
        gl_target = self.gl_target
        GL.glBindTexture(gl_target, t)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if data is None:
            data = pyopengl_null()
        dim = self.dimension
        if dim == 1:
            GL.glTexImage1D(gl_target, 0, iformat, size[0], 0, format, tdtype,
                            data)
        elif dim == 2:
            GL.glTexImage2D(gl_target, 0, iformat, size[0], size[1], 0, format,
                            tdtype, data)
        elif dim == 3:
            GL.glTexImage3D(gl_target, 0, iformat, size[0], size[1], size[2],
                            0, format, tdtype, data)

        GL.glTexParameterfv(gl_target, GL.GL_TEXTURE_BORDER_COLOR,
                            (0, 0, 0, 0))
        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_S,
                           GL.GL_CLAMP_TO_BORDER)
        if dim >= 2:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_T,
                               GL.GL_CLAMP_TO_BORDER)
        if dim >= 3:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_R,
                               GL.GL_CLAMP_TO_BORDER)

#        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_S,
#                           GL.GL_CLAMP_TO_EDGE)
#        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_T,
#                           GL.GL_CLAMP_TO_EDGE)
#        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_R,
#                           GL.GL_CLAMP_TO_EDGE)

#        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
#        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        if depth_compare_mode:
            # For GLSL sampler2dShadow objects to compare depth
            # to r texture coord.
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_COMPARE_MODE,
                               GL.GL_COMPARE_REF_TO_TEXTURE)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_COMPARE_FUNC,
                               GL.GL_LEQUAL)

        if ncomp == 1 or ncomp == 2:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
        if ncomp == 2:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_GREEN)
        GL.glBindTexture(gl_target, 0)

    def __del__(self):
        'Delete the OpenGL texture.'
        GL.glDeleteTextures((self.id,))

    def bind_texture(self, tex_unit=None):
        'Bind the OpenGL texture.'
        if tex_unit is None:
            GL.glBindTexture(self.gl_target, self.id)
        else:
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(self.gl_target, self.id)
            GL.glActiveTexture(GL.GL_TEXTURE0)

    def unbind_texture(self, tex_unit=None):
        'Unbind the OpenGL texture.'
        if tex_unit is None:
            GL.glBindTexture(self.gl_target, 0)
        else:
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
            GL.glBindTexture(self.gl_target, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)

    def reload_texture(self, data):
        '''
        Replace the texture values in texture with OpenGL id using numpy
        array data.  The data is interpreted the same as for the Texture
        constructor data argument.
        '''

        dim = self.dimension
        size = data.shape[dim - 1::-1]
        format, iformat, tdtype, ncomp = self.texture_format(data)
        gl_target = self.gl_target
        GL.glBindTexture(gl_target, self.id)
        if dim == 1:
            GL.glTexSubImage2D(gl_target, 0, 0, 0, size[0], format, tdtype,
                               data)
        elif dim == 2:
            GL.glTexSubImage2D(gl_target, 0, 0, 0, size[0], size[1], format,
                               tdtype, data)
        elif dim == 3:
            GL.glTexSubImage3D(gl_target, 0, 0, 0, size[0], size[1], size[2],
                               format, tdtype, data)
        GL.glBindTexture(gl_target, 0)

    def texture_format(self, data):
        '''
        Return the OpenGL texture format, internal format, and texture
        value type that will be used by the glTexImageNd() function when
        creating a texture from a numpy array of colors.
        '''
        dim = self.dimension
        if dim == 2 and len(data.shape) == dim and data.itemsize == 4:
            format = GL.GL_RGBA
            iformat = GL.GL_RGBA8
            tdtype = GL.GL_UNSIGNED_BYTE
            ncomp = 4
            return format, iformat, tdtype, ncomp

        ncomp = data.shape[dim] if len(data.shape) > dim else 1
        # TODO: Report pyopengl bug, GL_RG missing
        GL.GL_RG = 0x8227
        # luminance texture formats are not in opengl 3.
        format = {1: GL.GL_RED, 2: GL.GL_RG,
                  3: GL.GL_RGB, 4: GL.GL_RGBA}[ncomp]
        iformat = {1: GL.GL_RED, 2: GL.GL_RG,
                   3: GL.GL_RGB8, 4: GL.GL_RGBA8}[ncomp]
        dtype = data.dtype
        from numpy import uint8, float32
        if dtype == uint8:
            tdtype = GL.GL_UNSIGNED_BYTE
        elif dtype == float32:
            tdtype = GL.GL_FLOAT
        else:
            raise TypeError('Texture value type %s not supported' % str(dtype))
        return format, iformat, tdtype, ncomp


class TextureWindow:
    '''Draw a texture on a full window rectangle.'''
    def __init__(self, render, shader_options):

        # Must have vao bound before compiling shader.
        self.vao = vao = Bindings()
        vao.activate()

        p = render.shader(shader_options)
        render.use_shader(p)
        vao.shader = p

        self.vertex_buf = vb = Buffer(VERTEX_BUFFER)
        from numpy import array, float32, int32
        vb.update_buffer_data(array(((-1, -1, 0), (1, -1, 0), (1, 1, 0),
                                    (-1, 1, 0)), float32))
        vao.bind_shader_variable(vb)
        self.tex_coord_buf = tcb = Buffer(TEXTURE_COORDS_2D_BUFFER)
        tcb.update_buffer_data(array(((0, 0), (1, 0), (1, 1), (0, 1)),
                                     float32))
        vao.bind_shader_variable(tcb)
        self.element_buf = eb = Buffer(ELEMENT_BUFFER)
        eb.update_buffer_data(array(((0, 1, 2), (0, 2, 3)), int32))
        vao.bind_shader_variable(eb)    # Binds element buffer for rendering

    def __del__(self):
        self.vao = None
        for b in (self.vertex_buf, self.tex_coord_buf, self.element_buf):
            b.delete_buffer()

    def draw(self, xshift=0, yshift=0):
        xs, ys = xshift, yshift
        tcb = self.tex_coord_buf
        from numpy import array, float32
        tcb.update_buffer_data(array(((xs, ys), (1 + xs, ys), (1 + xs, 1 + ys),
                                     (xs, 1 + ys)), float32))
        GL.glDepthMask(False)   # Don't overwrite depth buffer
        eb = self.element_buf
        eb.draw_elements(eb.triangles)
        GL.glDepthMask(True)


def print_debug_log(tag, count=None):
    # GLuint glGetDebugMessageLog(GLuint count, GLsizei bufSize,
    #   GLenum *sources, Glenum *types, GLuint *ids, GLenum *severities,
    #   GLsizei *lengths, GLchar *messageLog)
    if count is None:
        while print_debug_log(tag, 1) > 0:
            continue
        return
    print('print_debug_log', GL.glIsEnabled(GL.GL_DEBUG_OUTPUT))
    buf = bytes(8192)
    sources = pyopengl_null()
    types = pyopengl_null()
    ids = pyopengl_null()
    severities = pyopengl_null()
    lengths = pyopengl_null()
    num_messages = GL.glGetDebugMessageLog(count, len(buf), sources, types,
                                           ids, severities, lengths, buf)
    if num_messages == 0:
        return 0
    print(tag, buf.decode('utf-8', 'replace'))
    return num_messages


def pyopengl_null():
    import ctypes
    return ctypes.c_void_p(0)

class OffScreenRenderingContext:
    def __init__(self, width = 512, height = 512):
        self.width = width
        self.height = height
        from OpenGL import GL, arrays, platform
        from OpenGL.raw.osmesa import mesa
        # To use OSMesa with PyOpenGL requires environment variable PYOPENGL_PLATFORM=osmesa
        # Also will need libOSMesa in dlopen library path.
        self.context = mesa.OSMesaCreateContext(GL.GL_RGBA, None)
        buf = arrays.GLubyteArray.zeros((height, width, 4))
        #p = arrays.ArrayDatatype.dataPointer(buf)
        self.buffer = buf
        
    def make_current(self):
        from OpenGL import GL, arrays, platform
        from OpenGL.raw.osmesa import mesa
        assert(mesa.OSMesaMakeCurrent(self.context, self.buffer, GL.GL_UNSIGNED_BYTE, self.width, self.height))
        assert(platform.CurrentContextIsValid())
        print('current osmesa context', platform.GetCurrentContext())

    def swap_buffers(self):
        pass
