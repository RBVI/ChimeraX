# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

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

# Set to PyOpenGL module OpenGL.GL by _initialize_pyopengl().
GL = None

# OpenGL workarounds:
stencil8_needed = False

class OpenGLVersionError(RuntimeError):
    pass

class OpenGLError(RuntimeError):
    pass

class OpenGLContext:
    '''
    OpenGL context used by View for drawing.
    This implementation uses Qt QOpenGLContext.
    '''

    required_opengl_version = (3, 3)
    required_opengl_core_profile = True

    def __init__(self, graphics_window, screen, use_stereo = False):
        _initialize_pyopengl() # Set global GL module.

        self.window = graphics_window
        self._screen = screen
        self._context_thread = None
        self._color_bits = None          # None, 8, 12, 16
        self._depth_bits = 24
        self._framebuffer_color_bits = 8 # For offscreen framebuffers, 8 or 16
        self._mode = 'stereo' if use_stereo else 'mono'
        self._contexts = {}              # Map mode to QOpenGLContext, or False if creation failed
        self._share_context = None       # First QOpenGLContext, shares state with others
        self._create_failed = False
        self._wait_for_vsync = True
        self._deleted = False

        # Keep track of allocated framebuffers and vertex array objects
        # so we can release those when switching to a quad-buffered stereo context.
        from weakref import WeakSet
        self._framebuffers = WeakSet() # Set of Framebuffer objects
        self._bindings = WeakSet()     # Set of Bindings objects

        # Draw target for default framebuffer
        self.default_draw_target = GL.GL_BACK

    def __del__(self):
        if not self._deleted:
            self.delete()

    def delete(self):
        self._deleted = True
        from Qt import qt_object_is_deleted
        for oc in self._contexts.values():
            if oc and not qt_object_is_deleted(oc):
                oc.deleteLater()
        self._contexts.clear()
        self._share_context = None

    @property
    def _qopengl_context(self):
        return self._contexts.get(self._mode)

    @property
    def created(self):
        return self._qopengl_context is not None

    def make_current(self, window = None):
        '''Make the OpenGL context active.'''
        if self._create_failed:
            return False

        qc = self._qopengl_context
        if qc is None:
            # create context
            try:
                qc = self._initialize_context()
            except (OpenGLError, OpenGLVersionError):
                self._create_failed = True
                raise

        self._check_thread()

        w = self.window if window is None else window
        if not qc.makeCurrent(w):
            raise OpenGLError("Could not make graphics context current")

        return True

    def _initialize_context(self, mode = None, window = None):
        '''Can raise OpenGLError, or OpenGLVersion error.'''
        if mode is None:
            mode = self._mode

        if window is None:
            window = self.window

        # Remember thread where context is valid
        self._set_thread()

        # Create context
        from Qt.QtGui import QOpenGLContext
        qc = QOpenGLContext()

        # Use screen window is on if it has been mapped.
        screen = window.screen()
        if screen is None:
            self._screen
        qc.setScreen(screen)

        if self._share_context:
            qc.setShareContext(self._share_context)

        # Set OpenGL context format
        fmt = self._context_format(mode)
        qc.setFormat(fmt)
        window.setFormat(fmt)

        # Validate context
        if not qc.create():
            self._contexts[mode] = False
            raise OpenGLError("Could not create OpenGL context")

        # Check if stereo obtained.
        got_fmt = qc.format()
        if fmt.stereo() and not got_fmt.stereo():
            raise OpenGLError("Could not create stereo OpenGL context")

        # Check if opengl version is adequate.
        try:
            self._check_context_version(qc.format())
        except Exception:
            self._contexts[mode] = False
            raise # OpenGLVersionError

        if window:
            self.window = window
        if self._share_context is None:
            self._share_context = qc
        self._contexts[mode] = qc
        self._mode = mode
        return qc

    def _context_format(self, mode):
        from Qt.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setVersion(*self.required_opengl_version)
        cbits = self._color_bits
        if cbits is not None:
            fmt.setRedBufferSize(cbits)
            fmt.setGreenBufferSize(cbits)
            fmt.setBlueBufferSize(cbits)
        fmt.setDepthBufferSize(self._depth_bits)
        if self.required_opengl_core_profile:
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        if not self._wait_for_vsync:
            # Don't wait for vsync.
            # Works on Mac OS 10.13 Nvidia graphics with Qt 5.
            # Works on Windows 10, Nvidia P6000 with Qt 6.5.
            fmt.setSwapInterval(0)
        if mode == 'stereo':
            fmt.setStereo(True)
        return fmt

    def _check_context_version(self, fmt):
        major, minor = fmt.version()
        rmajor, rminor = self.required_opengl_version
        if major < rmajor or (major == rmajor and minor < rminor):
            from chimerax.graphics import OpenGLVersionError
            raise OpenGLVersionError(
                'ChimeraX requires OpenGL graphics version %d.%d.\n' % (rmajor, rminor) +
                'Your computer graphics driver provided version %d.%d\n' % (major, minor) +
                'Try updating your graphics driver.')
        if self.required_opengl_core_profile:
            if fmt.profile() != fmt.OpenGLContextProfile.CoreProfile:
                from chimerax.graphics import OpenGLVersionError
                raise OpenGLVersionError(
                    'ChimeraX requires an OpenGL graphics core profile.\n' +
                    'Your computer graphics driver is only offering a non-core profile (version %d.%d).\n'
                    % (major, minor) +
                    'Try updating your graphics driver.')

    def _set_thread(self):
        # Remember the thread context was created in.
        # Can only use context in this thread, otherwise makeCurrent crashes.
        ct = self._context_thread
        import threading
        t = threading.get_ident()
        if ct is None:
            self._context_thread = t
        elif t != ct:
            raise RuntimeError('Attempted to create OpenGLContext in wrong thread (%s, previous thread %s).'
                               % (t, ct))

    def _check_thread(self):
        ct = self._context_thread
        import threading
        t = threading.get_ident()
        if t != ct:
            raise RuntimeError('Attempted to make OpenGL context current in wrong thread (%s, context thread %s).'
                               % (t, ct))

    def done_current(self):
        '''Makes no context current.'''
        self._qopengl_context.doneCurrent()

    def swap_buffers(self, window = None):
        '''Swap back and front OpenGL buffers.'''
        w = self.window if window is None else window
        self._qopengl_context.swapBuffers(w)

    def swap_interval(self):
        from OpenGL.WGL.EXT.swap_control import wglGetSwapIntervalEXT
        return wglGetSwapIntervalEXT()

    def wait_for_vsync(self, wait):
        '''
        Control whether OpenGL synchronizes to the display vertical refresh.
        Currently this call is only supported on Windows. Returns true if
        the setting can be changed, otherwise false.
        '''
        results = False
        from sys import platform
        if platform == 'win32':
            wfmt = self.window.format()
            wfmt.setSwapInterval(1 if wait else 0)
            self.window.setFormat(wfmt)
            self._wait_for_vsync = wait
            success = True
            # succes = _set_windows_swap_interval(wait)
        elif platform == 'darwin':
            sync = 1 if wait else 0
            from ._graphics import set_mac_swap_interval
            success = set_mac_swap_interval(sync)
        elif platform == 'linux':
            sync = 1 if wait else 0
            from ._graphics import set_linux_swap_interval
            success = set_linux_swap_interval(sync)

        return success

    def pixel_scale(self):
        '''
        Ratio window toolkit pixel size to OpenGL pixel size.
        Usually 1, but 2 for Mac retina displays.
        '''
        return self.window.devicePixelRatio()

    @property
    def stereo(self):
        return self._mode == 'stereo'

    def enable_stereo(self, stereo, window):
        mode = 'stereo' if stereo else 'mono'
        if mode == self._mode:
            return True

        if len(self._contexts) == 0:
            self._mode = 'stereo' if stereo else 'mono'
            return True

        # Delete framebuffers and vertex array objects since those cannot be shared.
        self.make_current()
        for fb in tuple(self._framebuffers):
            fb._release()
        for bi in tuple(self._bindings):
            bi._release()
        self.current_shader_program = None
        self.current_viewport = None
        self.done_current()

        # Replace current context with stereo context while sharing opengl state.
        qc = self._contexts.get(mode)
        if not qc:
            # This can raise OpenGLError
            qc = self._initialize_context(mode, window)

        self._mode = mode
        self.window = window

        import sys
        if sys.platform == 'linux':
            # On Linux get GL_BACK_RIGHT error after switching
            # into stereo a second time.  Bug #2446.  To work around
            # this delete the stereo context after switching to mono.
            if not stereo:
                sqc = self._contexts.get('stereo')
                if sqc:
                    del self._contexts['stereo']
                    sqc.deleteLater()

    def set_offscreen_color_bits(self, bits):
        cbits = self._framebuffer_color_bits
        if bits != cbits:
            self._framebuffer_color_bits = bits
            self.make_current()
            for fb in tuple(self._framebuffers):
                fb.set_color_bits(bits)
            self.done_current()

def _set_windows_swap_interval(wait):
    # Qt 6 on Windows overrides the swap interval each time makeCurrent()
    # is called.  So this code is not useful and instead it is necessary
    # to change the QSurfaceFormat assigned to the QWindow.
    try:
        from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT
        i = 1 if wait else 0
        success = wglSwapIntervalEXT(i)
        success = True if success else False
    except Exception:
        success = False
    return success

_initialized_pyopengl = False
def _initialize_pyopengl(log_opengl_calls = False, offscreen = False):
    global _initialized_pyopengl
    if _initialized_pyopengl:
        return
    _initialized_pyopengl = True

    if offscreen:
        _configure_pyopengl_to_use_osmesa()

    if log_opengl_calls:
        # Log all OpenGL calls
        import logging
        from os.path import expanduser
        logging.basicConfig(level=logging.DEBUG, filename=expanduser('~/Desktop/cx.log'))
        logging.info('started logging')
        import OpenGL
        OpenGL.FULL_LOGGING = True

    #import OpenGL
    #OpenGL.ERROR_CHECKING = False

    global GL
    import OpenGL.GL
    GL = OpenGL.GL

def _configure_pyopengl_to_use_osmesa():
    '''Tell PyOpenGL where to find libOSMesa.'''

    # Get libOSMesa from the Python module osmesa if it exists.
    try:
        import osmesa
    except ImportError:
        # Let PyOpenGL try to find a system libOSMesa library.
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        return

    # PyOpenGL 3.1.5 can only find libOSMesa in system locations.
    # This hack allows it to find libOSMesa in the Python osmesa module.
    from OpenGL.platform.osmesa import OSMesaPlatform
    import ctypes
    OSMesaPlatform.GL = ctypes.CDLL(osmesa.osmesa_library_path(), ctypes.RTLD_GLOBAL)

    import os
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    # Reload the PyOpenGL platform which is set when OpenGL first imported.
    from OpenGL.platform import _load
    _load()

def remember_current_opengl_context():
    '''
    Return an object that notes the current opengl context and its window
    so it can later be restored by restore_current_opengl_context().
    '''
    from Qt.QtGui import QOpenGLContext
    opengl_context = QOpenGLContext.currentContext()
    opengl_surface = opengl_context.surface() if opengl_context else None
    return (opengl_context, opengl_surface)

def restore_current_opengl_context(remembered_context):
    '''
    Make the opengl context and window returned by remember_current_opengl_context()
    the current context.
    '''
    opengl_context, opengl_surface = remembered_context
    from Qt.QtGui import QOpenGLContext
    if opengl_context and QOpenGLContext.currentContext() != opengl_context:
        opengl_context.makeCurrent(opengl_surface)

class Render:
    '''
    Manage shaders, viewing matrices and lighting parameters to render a scene.
    '''
    def __init__(self, opengl_context):

        self._opengl_context = oc = opengl_context
        self._recording_calls = None
        self._front_buffer_valid = False

        if not hasattr(oc, 'shader_programs'):
            oc.shader_programs = {}
            oc.current_shader_program = None
            oc.current_viewport = None

        self.enable_capabilities = 0    # Bit field of SHADER_* capabilities
        self.disable_capabilities = 0

        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        # Used for optimizing model view matrix updates:
        self.current_model_matrix = None
        # Maps scene to camera coordinates:
        self.current_view_matrix = None
        self._near_far_clip = (0,1)             # Scene coord distances from eye
        self._clip_planes = []                  # Up to 8 4-tuples
        self._num_enabled_clip_planes = 0

        self.lighting = Lighting()
        self._lighting_buffer = None          # Uniform buffer for lighting parameters
        self._lighting_block = 1              # Uniform block binding point
        self._lighting_buffer_parameters = (  # Shader parameter name and float offset
            ("key_light_direction",0), ("key_light_diffuse_color",4),
            ("key_light_specular_color",8),  ("key_light_specular_exponent",11),
            ("fill_light_direction",12), ("fill_light_diffuse_color",16),
            ("ambient_color",20))
        self._lighting_buffer_floats = 28

        self.material = Material()              # Currently a global material

        self._default_framebuffer = None
        self.framebuffer_stack = [self.default_framebuffer()]

        self._texture_win = None

        # 3D ambient texture transform from model coordinates to texture
        # coordinates:
        self.ambient_texture_transform = None

        # Key light shadow
        self.shadow = Shadow(self, texture_unit = 1)

        # Multishadow
        self.multishadow = Multishadow(self, texture_unit = 2)

        # Silhouette edges
        self.silhouette = Silhouette()

        # Selection outlines
        self.outline = Outline(self)
        self._last_background_color = (0,0,0,1)   # RGBA 0-1 float scale

        # Offscreen rendering. Used for 16-bit color depth.
        self.offscreen = Offscreen()

        # Blending textures for multichannel image rendering.
        self.blend = BlendTextures(self)

        self.model_color = (1, 1, 1, 1)
        self._colormap_texture_unit = 4

        # Depth texture rendering parameters
        self.depth_texture_unit = 3
        self._depth_texture_scale = None # texture xscale, yscale and value zscale

        self.frame_number = 0

        # Camera origin, y, and xshift for SHADER_STEREO_360 mode
        self._stereo_360_params = ((0,0,0),(0,1,0),0)

        # OpenGL texture size limit
        self._max_3d_texture_size = None
        
    def delete(self):
        if self._opengl_context._deleted:
            raise RuntimeError('Render.delete(): OpenGL context deleted before Render instance')
        elif self._opengl_context.created:
            self.make_current()

        fb = self._default_framebuffer
        if fb:
            fb.delete()
            self._default_framebuffer = None

        lb = self._lighting_buffer
        if lb is not None:
            GL.glDeleteBuffers(1, [lb])
            self._lighting_buffer = None

        tw = self._texture_win
        if tw is not None:
            tw.delete()
            self._texture_win = None

        self.shadow.delete()
        self.shadow = None

        self.multishadow.delete()
        self.multishadow = None

        self.silhouette.delete()
        self.silhouette = None

        self.outline.delete()
        self.outline = None

        self.offscreen.delete()
        self.offscreen = None

        self.blend.delete()
        self.blend = None

    @property
    def opengl_context(self):
        return self._opengl_context

    def make_current(self):
        return self._opengl_context.make_current()

    def done_current(self):
        self._opengl_context.done_current()

    def swap_buffers(self):
        self._opengl_context.swap_buffers()
        self._front_buffer_valid = True

    @property
    def front_buffer_valid(self):
        return self._front_buffer_valid

    def swap_interval(self):
        return self._opengl_context.swap_interval()

    def wait_for_vsync(self, wait):
        return self._opengl_context.wait_for_vsync(wait)

    def use_shared_context(self, window):
        '''
        Switch opengl context to use the specified target QWindow.
        Multiple Render instances can share the same opengl context
        using this method.
        '''
        oc = self._opengl_context
        prev_win = oc.window
        oc.window = window
        try:
            self.make_current()
            s = oc.pixel_scale()
            self.set_viewport(0,0,int(s*window.width()),int(s*window.height()))
        except:
            # Make sure to restore opengl context window if make_current() fails.
            # This probably indicates the window has been destroyed.  Bug #3605.
            oc.window = prev_win
            raise
        if window != prev_win:
            self._front_buffer_valid = False
        return prev_win

    @property
    def recording_opengl(self):
        return self._recording_calls is not None

    def record_opengl_calls(self, record = True):
        if record:
            from . import gllist
            self._recording_calls = rc = gllist.start_gl_call_list()
            globals()['GL'] = gllist
            return rc
        else:
            from OpenGL import GL
            globals()['GL'] = GL
            self._recording_calls = None

    @property
    def current_shader_program(self):
        return self._opengl_context.current_shader_program

    def _get_current_viewport(self):
        return self._opengl_context.current_viewport
    def _set_current_viewport(self, xywh):
        self._opengl_context.current_viewport = xywh
    current_viewport = property(_get_current_viewport, _set_current_viewport)

    def default_framebuffer(self):
        if self._default_framebuffer is None:
            self._default_framebuffer = Framebuffer('default', self.opengl_context, color=False, depth=False)
        return self._default_framebuffer

    def set_default_framebuffer_size(self, width, height):
        '''
        This does not update the viewport.
        Use Render.update_viewport() when OpenGL context current to update viewport
        '''
        s = self._opengl_context.pixel_scale()
        w, h = int(s*width), int(s*height)
        fb = self.default_framebuffer()
        fb.width, fb.height = w, h
        fb.viewport = (0, 0, w, h)

    def render_size(self):
        fb = self.current_framebuffer()
        x, y, w, h = fb.viewport
        return (w, h)

    def max_framebuffer_size(self):
        max_rb_size = GL.glGetInteger(GL.GL_MAX_RENDERBUFFER_SIZE)
        max_tex_size = GL.glGetInteger(GL.GL_MAX_TEXTURE_SIZE)
        max_size = min(max_rb_size, max_tex_size)
        return max_size

    def max_3d_texture_size(self):
        if self._max_3d_texture_size is None:
            self._max_3d_texture_size = GL.glGetInteger(GL.GL_MAX_3D_TEXTURE_SIZE)
        return self._max_3d_texture_size
    
    def framebuffer_rgba_bits(self):
        # This is only valid for default framebuffer.
        # Need to use GL_COLOR_ATTACHMENT0 for offscreen framebuffers.
        return tuple(GL.glGetFramebufferAttachmentParameteriv(GL.GL_DRAW_FRAMEBUFFER,
                                                              GL.GL_BACK_LEFT, attr)
                     for attr in (GL.GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE,
                                  GL.GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE,
                                  GL.GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE,
                                  GL.GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE))

    def framebuffer_depth_bits(self):
        # This is only valid for default framebuffer.
        # Need to use GL_DEPTH_ATTACHMENT for offscreen framebuffers.
        return GL.glGetFramebufferAttachmentParameteriv(GL.GL_DRAW_FRAMEBUFFER,
                                                        GL.GL_DEPTH,
                                                        GL.GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE)

    def set_offscreen_color_bits(self, bits):
        self._opengl_context.set_offscreen_color_bits(bits)

    def disable_shader_capabilities(self, ocap):
        self.disable_capabilities = ocap

    def draw_depth_only(self, depth_only=True):
        # Enable only shader geometry, no colors or lighting.
        if depth_only:
            d = ~(self.SHADER_INSTANCING | self.SHADER_SHIFT_AND_SCALE |
                  self.SHADER_TRANSPARENT_ONLY | self.SHADER_OPAQUE_ONLY |
                  self.SHADER_CLIP_PLANES)
        else:
            d = 0
        self.disable_capabilities = d
        c = GL.GL_FALSE if depth_only else GL.GL_TRUE
        GL.glColorMask(c, c, c, c)

    def shader(self, options):
        '''
        Return a shader that supports the specified capabilities.
        Also activate the shader with glUseProgram().
        The capabilities are specified as at bit field of values from
        SHADER_LIGHTING, SHADER_DEPTH_CUE, SHADER_TEXTURE_2D, SHADER_TEXTURE_3D,
        SHADER_COLORMAP, SHADER_DEPTH_TEXTURE, SHADER_TEXTURE_CUBEMAP,
        SHADER_TEXTURE_3D_AMBIENT, SHADER_SHADOW, SHADER_MULTISHADOW,
        SHADER_SHIFT_AND_SCALE, SHADER_INSTANCING, SHADER_TEXTURE_OUTLINE,
        SHADER_DEPTH_OUTLINE, SHADER_VERTEX_COLORS,
        SHADER_TRANSPARENT_ONLY, SHADER_OPAQUE_ONLY, SHADER_STEREO_360
        SHADER_CLIP_PLANES, SHADER_ALL_WHITE
        '''
        options |= self.enable_capabilities
        options &= ~self.disable_capabilities
        p = self._opengl_shader(options)
        return p

    def _use_shader(self, shader):
        '''
        Set the current shader.
        '''
        if shader == self.current_shader_program:
            return

        # print('changed shader', ', '.join(shader_capability_names(shader.capabilities)))
        self._opengl_context.current_shader_program = shader
        c = shader.capabilities
        GL.glUseProgram(shader.program_id)
        if self.SHADER_LIGHTING & c:
            if self.SHADER_TEXTURE_3D_AMBIENT & c:
                shader.set_integer('tex3d', 0)    # Tex unit 0.
            if self.SHADER_MULTISHADOW & c:
                self.multishadow._set_multishadow_shader_variables(shader)
            if self.SHADER_SHADOW & c:
                self.shadow._set_shadow_shader_variables(shader)
        if self.SHADER_DEPTH_CUE & c:
            self.set_depth_cue_parameters()
        if not (self.SHADER_TEXTURE_OUTLINE & c
                or self.SHADER_DEPTH_OUTLINE & c
                or self.SHADER_BLEND_TEXTURE_2D & c
                or self.SHADER_BLEND_TEXTURE_3D & c):
            self.set_projection_matrix()
            self.set_model_matrix()
        if (self.SHADER_TEXTURE_2D & c
            or self.SHADER_TEXTURE_OUTLINE & c
            or self.SHADER_DEPTH_OUTLINE & c
            or self.SHADER_BLEND_TEXTURE_2D & c):
            shader.set_integer("tex2d", 0)    # Texture unit 0.
        if (self.SHADER_TEXTURE_3D & c
            or self.SHADER_BLEND_TEXTURE_3D & c):
            shader.set_integer("tex3d", 0)
        if self.SHADER_COLORMAP & c or self.SHADER_BLEND_COLORMAP & c:
            shader.set_integer("colormap", self._colormap_texture_unit)
        if self.SHADER_DEPTH_TEXTURE & c:
            self._set_depth_texture_shader_variables(shader)
        if self.SHADER_TEXTURE_CUBEMAP & c:
            shader.set_integer("texcube", 0)
        if not self.SHADER_VERTEX_COLORS & c:
            self.set_model_color()
        if self.SHADER_FRAME_NUMBER & c:
            self.set_frame_number()
        if self.SHADER_STEREO_360 & c:
            self.set_stereo_360_params()
        self.set_clip_parameters()
        shader.validate_program() # Check that OpenGL is setup right (only happens one time).

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
            raise OpenGLError('No framebuffer left on stack.')
        fb = s[-1]
        fb.activate()
        self.set_viewport(*fb.viewport)
        return pfb

    def rendering_to_screen(self):
        return len(self.framebuffer_stack) == 1

    def _opengl_shader(self, capabilities):
        'Return OpenGL shader program id, creating shader if needed.'

        p = None
        sp = self._opengl_context.shader_programs
        if capabilities in sp:
            p = sp[capabilities]
        else:
            # Shadow or depth cue off overrides on.
            # On is usually a global setting where off is per-drawing.
            orig_cap = capabilities
            cap_pairs = ((self.SHADER_NO_SHADOW, self.SHADER_SHADOW),
                         (self.SHADER_NO_MULTISHADOW, self.SHADER_MULTISHADOW),
                         (self.SHADER_NO_DEPTH_CUE, self.SHADER_DEPTH_CUE),
                         (self.SHADER_NO_CLIP_PLANES, self.SHADER_CLIP_PLANES))
            for nc, c in cap_pairs:
                if capabilities & nc:
                    capabilities &= ~(c | nc)
            if capabilities in sp:
                p = sp[capabilities]
                sp[orig_cap] = p
        if p is None:
            p = Shader(capabilities = capabilities, max_shadows = self.multishadow.max_multishadows())
            sp[capabilities] = p
            if capabilities & self.SHADER_LIGHTING:
                self._bind_lighting_parameter_buffer(p)
                if capabilities & self.SHADER_MULTISHADOW:
                    GL.glUseProgram(p.program_id)
                    self.multishadow._set_multishadow_shader_constants(p)

        self._use_shader(p)
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
        if p is not None and not (p.capabilities & self.SHADER_NO_PROJECTION_MATRIX):
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
            mv = self.current_model_view_matrix
            if mv is None:
                return
        else:
            # TODO: optimize check of same model matrix.
            cmm = self.current_model_matrix
            if cmm:
                if ((model_matrix.is_identity() and cmm.is_identity())
                        or model_matrix.same(cmm)):
                    return
            self.current_model_matrix = model_matrix
            # TODO: optimize matrix multiply.  Rendering bottleneck with 200 models open.
            cvm = self.current_view_matrix
            mv = cvm if model_matrix.is_identity() else (cvm * model_matrix)
            self.current_model_view_matrix = mv

        p = self.current_shader_program
        if (p is not None and
            not p.capabilities & self.SHADER_TEXTURE_OUTLINE and
            not p.capabilities & self.SHADER_DEPTH_OUTLINE):
            if (p.capabilities & self.SHADER_LIGHTING or
                not p.capabilities & self.SHADER_STEREO_360):
                p.set_matrix('model_view_matrix', mv.opengl_matrix())
#            if (self.SHADER_CLIP_PLANES | self.SHADER_MULTISHADOW) & p.capabilities:
            if self.SHADER_CLIP_PLANES & p.capabilities:
                cmm = self.current_model_matrix
                if cmm:
                    p.set_matrix('model_matrix', cmm.opengl_matrix())
                    self.set_clip_parameters()
            if self.SHADER_STEREO_360 & p.capabilities:
                cmm = self.current_model_matrix
                cvm = self.current_view_matrix
                if cmm:
                    p.set_matrix('model_matrix', cmm.opengl_matrix())
                if cvm:
                    p.set_matrix('view_matrix', cvm.opengl_matrix())
            if not self.lighting.move_lights_with_camera:
                self.update_lighting_parameters()

    def set_near_far_clip(self, near_far):
        '''Set the near and far clip plane distances from eye.  Used for depth cuing.'''
        self._near_far_clip = near_far

        p = self.current_shader_program
        if p is not None and p.capabilities & self.SHADER_DEPTH_CUE:
            self.set_depth_cue_parameters()

    def set_clip_parameters(self, clip_planes = None):
        if clip_planes is not None:
            self._clip_planes = clip_planes

        p = self.current_shader_program
        if p is None:
            return

        m = self.current_model_matrix
        cp = self._clip_planes
        if self.SHADER_CLIP_PLANES & p.capabilities and m is not None and cp:
            p.set_matrix('model_matrix', m.opengl_matrix())
            p.set_integer('num_clip_planes', len(cp))
            p.set_float4('clip_planes', cp, len(cp))
            for i in range(len(cp)):
                GL.glEnable(GL.GL_CLIP_DISTANCE0 + i)
            for i in range(len(cp), self._num_enabled_clip_planes):
                GL.glDisable(GL.GL_CLIP_DISTANCE0 + i)
            self._num_enabled_clip_planes = len(cp)
        else:
            for i in range(self._num_enabled_clip_planes):
                GL.glDisable(GL.GL_CLIP_DISTANCE0 + i)
            self._num_enabled_clip_planes = 0

    def set_frame_number(self, f=None):
        if f is None:
            f = self.frame_number
        else:
            self.frame_number = f
        p = self.current_shader_program
        if p is not None and self.SHADER_FRAME_NUMBER & p.capabilities:
            p.set_float('frame_number', f)

    def set_lighting_shader_capabilities(self):
        lp = self.lighting
        self.enable_shader_capability(self.SHADER_DEPTH_CUE, lp.depth_cue)
        self.enable_shader_shadows(lp.shadows)
        self.enable_shader_multishadows(lp.multishadow > 0)

    def enable_shader_shadows(self, enable):
        self.enable_shader_capability(self.SHADER_SHADOW, enable)

    def enable_shader_multishadows(self, enable):
        self.enable_shader_capability(self.SHADER_MULTISHADOW, enable)

    def enable_shader_capability(self, capability, enable):
        if enable:
            self.enable_capabilities |= capability
        else:
            self.enable_capabilities &= ~capability

    def _bind_lighting_parameter_buffer(self, shader):
        pid = shader.program_id
        bi = GL.glGetUniformBlockIndex(pid, b'lighting_block')
        GL.glUniformBlockBinding(pid, bi, self._lighting_block)

    def _lighting_parameter_buffer(self):
        b = self._lighting_buffer
        if b is not None:
            return b

        # Create uniform buffer for lighting parameters, shared by all shaders.
        self._lighting_buffer = b = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, b)
        nbytes = 4*self._lighting_buffer_floats
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, nbytes, pyopengl_null(), GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

        return b

    def activate_lighting(self):
        # If two Render instances are rendering different lighting this is needed to switch
        # between their lighting buffers.
        # TODO: Optimize case with one Render instance to not update lighting binding every frame.
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, self._lighting_block, self._lighting_buffer)

    def update_lighting_parameters(self):
        self._fill_lighting_parameter_buffer()

    def _fill_lighting_parameter_buffer(self):
        b = self._lighting_parameter_buffer()
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, b)
        offset = 0
        data = self._lighting_parameter_array()
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, offset, data.nbytes, data)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def _lighting_parameter_array(self):
        lparam = self._light_shader_parameter_values()
        size = self._lighting_buffer_floats
        from numpy import zeros, float32
        data = zeros((size,), float32)
        for name, offset in self._lighting_buffer_parameters:
            value = lparam[name]
            from numpy import ndarray
            if isinstance(value, (tuple, list, ndarray)):
                data[offset:offset+len(value)] = value
            else:
                data[offset] = value
        return data

    def _light_shader_parameter_values(self):
        lp = self.lighting
        mp = self.material

        move = None if lp.move_lights_with_camera else self.current_view_matrix

        params = {}

        # Key light
        from chimerax.geometry import normalize_vector
        kld = normalize_vector(lp.key_light_direction)
        if move:
            kld = move.transform_vector(kld)
        params["key_light_direction"] = kld
        ds = mp.diffuse_reflectivity * lp.key_light_intensity
        kdc = tuple(ds * c for c in lp.key_light_color)
        params["key_light_diffuse_color"] = kdc

        # Key light specular
        ss = mp.specular_reflectivity * lp.key_light_intensity
        ksc = tuple(ss * c for c in lp.key_light_color)
        params["key_light_specular_color"] = ksc
        params["key_light_specular_exponent"] = mp.specular_exponent

        # Fill light
        fld = normalize_vector(lp.fill_light_direction)
        if move:
            fld = move.transform_vector(fld)
        params["fill_light_direction"] = fld
        ds = mp.diffuse_reflectivity * lp.fill_light_intensity
        fdc = tuple(ds * c for c in lp.fill_light_color)
        params["fill_light_diffuse_color"] = fdc

        # Ambient light
        ams = mp.ambient_reflectivity * lp.ambient_light_intensity
        ac = tuple(ams * c for c in lp.ambient_light_color)
        params["ambient_color"] = ac

        return params

    def set_depth_cue_parameters(self):
        '''Private. Sets shader depth variables using the lighting
        parameters object given in the contructor.'''

        p = self.current_shader_program
        if p is None:
            return

        if self.SHADER_DEPTH_CUE & p.capabilities:
            if self.recording_opengl:
                r = lambda: self._depth_cue_range(self._near_far_clip())
            else:
                r = self._depth_cue_range(self._near_far_clip)
            p.set_vector2('depth_cue_range', r)
            p.set_vector3('depth_cue_color', self.lighting.depth_cue_color)

    def _depth_cue_range(self, near_far):
        lp = self.lighting
        n,f = near_far
        return (n + (f-n)*lp.depth_cue_start,
                n + (f-n)*lp.depth_cue_end)

    def set_model_color(self, color=None):
        '''
        Set the OpenGL shader color for shader single color mode.
        '''
        if color is not None:
            self.model_color = color
        p = self.current_shader_program
        if p is not None:
            if not ((self.SHADER_VERTEX_COLORS | self.SHADER_ALL_WHITE) & p.capabilities):
                p.set_rgba("color", self.model_color)

    def set_ambient_texture_transform(self, tf):
        # Transform from model coordinates to ambient texture coordinates.
        p = self.current_shader_program
        if p is not None:
            p.set_matrix("ambient_tex3d_transform", tf.opengl_matrix())

    def set_colormap(self, colormap_texture, colormap_range, data_texture):
        p = self.current_shader_program
        if p is None:
            raise OpenGLError('Render.set_colormap(): No current shader program.')
        if self.SHADER_COLORMAP & p.capabilities:
            colormap_texture.bind_texture(tex_unit = self._colormap_texture_unit)
            v0,v1 = colormap_range
            s = data_texture.normalization()
            p.set_vector2('colormap_range', (s*v0, 1/(s*(v1-v0))))

    def check_for_opengl_errors(self):
        # Clear previous errors.
        lines = []
        while True:
            e = GL.glGetError()
            if e == GL.GL_NO_ERROR:
                break
            from OpenGL import GLU
            es = GLU.gluErrorString(e)
            lines.append('OpenGL error %s' % es.decode('iso-8859-1'))
        msg = '\n'.join(lines)
        return msg

    def opengl_version(self):
        'String description of the OpenGL version for the current context.'
        return GL.glGetString(GL.GL_VERSION).decode('utf-8')

    def opengl_version_number(self):
        'Return major and minor opengl version numbers (integers).'
        vs = self.opengl_version().split()[0].split('.')[:2]
        vmajor, vminor = [int(v) for v in vs]
        return vmajor, vminor

    def opengl_vendor(self):
        'String description of the OpenGL vendor for the current context.'
        return GL.glGetString(GL.GL_VENDOR).decode('utf-8')

    def opengl_renderer(self):
        'String description of the OpenGL renderer for the current context.'
        return GL.glGetString(GL.GL_RENDERER).decode('utf-8')

    def check_opengl_version(self, major = 3, minor = 3):
        '''Check if current OpenGL context meets minimum required version.'''
        vmajor, vminor = self.opengl_version_number()
        if vmajor < major or (vmajor == major and vminor < minor):
            raise OpenGLVersionError('ChimeraX requires OpenGL graphics version 3.3.\n'
                                     'Your computer graphics driver provided version %d.%d.\n'
                                     % (vmajor, vminor))

    def opengl_info(self):
        lines = ['vendor: %s' % GL.glGetString(GL.GL_VENDOR).decode('utf-8'),
                 'renderer: %s' % GL.glGetString(GL.GL_RENDERER).decode('utf-8'),
                 'version: %s' % GL.glGetString(GL.GL_VERSION).decode('utf-8'),
                 'GLSL version: %s' % GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode('utf-8'),
                 'rgba bits: %d,%d,%d,%d' % self.framebuffer_rgba_bits(),
                 'depth bits: %d' % self.framebuffer_depth_bits()]
        ne = GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)
        for e in range(ne):
            lines.append('extension: %s' % GL.glGetStringi(GL.GL_EXTENSIONS,e).decode('utf-8'))
        return '\n'.join(lines)

    def opengl_profile(self):
        pmask = GL.glGetIntegerv(GL.GL_CONTEXT_PROFILE_MASK)
        if pmask == GL.GL_CONTEXT_CORE_PROFILE_BIT:
            p = 'core'
        elif pmask == GL.GL_CONTEXT_COMPATIBILITY_PROFILE_BIT:
            p = 'compatibility'
        else:
            p = 'unknown'
        return p

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

        s = self._opengl_context.pixel_scale()
        w, h = int(s*width), int(s*height)
        fb = self.default_framebuffer()
        fb.width, fb.height = w, h
        self.set_viewport(0, 0, w, h)

        self.enable_depth_test(True)

        # Detect OpenGL workarounds
        vendor = GL.glGetString(GL.GL_VENDOR)
        import sys
        global stencil8_needed
        stencil8_needed = (sys.platform.startswith('linux') and vendor and
                           vendor.startswith((b'AMD', b'ATI')))

        #
        # Outline drawing uses glBlitFramebuffer() to copy the depth buffer and
        # that fails if the default framebuffer depth attachment is not 24 bits
        # and the outlining depth is 24 bits.  In this case render offscreen so
        # both the render and outline framebuffers have 24-bit depth buffers.
        #
        # On macOS 10.14.5 with Radeon Pro Vega 20 or 16 graphics selection outlines
        # are fragmented if rendering is to the default framebuffer.  The problem
        # appears to be that copying the default framebuffer depth to the offscreen
        # mask framebuffer does not work correctly.  So use offscreen rendering
        # in this case.  ChimeraX bug #2216.
        #
        offscreen_outline = False
        if self.framebuffer_depth_bits() != 24:
            offscreen_outline = True
        elif sys.platform.startswith('darwin'):
            rname = self.opengl_renderer()
            # Bugs in Apple AMD graphics drivers
            if (rname.startswith('AMD Radeon Pro Vega') or
                rname.startswith('AMD Radeon Pro 5500M') or # ChimeraX bug 4238
                rname.startswith('AMD Radeon Pro 5300M') or # ChimeraX bug 6014
                rname.startswith('AMD Radeon Pro 5700')): # ChimeraX bug 7585
                offscreen_outline = True
        self.outline.offscreen_outline_needed = offscreen_outline

    def pixel_scale(self):
        return self._opengl_context.pixel_scale()

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

    def update_viewport(self):
        'Update viewport if framebuffer viewport different from current viewport.'
        fb = self.current_framebuffer()
        if fb and fb.viewport != self.current_viewport:
            self.set_viewport(*fb.viewport)

    def set_background_color(self, rgba):
        'Set the OpenGL clear color.'
        r, g, b, a = rgba
        GL.glClearColor(r, g, b, a)
        self._last_background_color = tuple(rgba)

    def draw_background(self, depth=True):
        'Draw the background color and clear the depth buffer.'
        flags = (GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT) if depth else GL.GL_COLOR_BUFFER_BIT
        GL.glClear(flags)

    def enable_depth_test(self, enable):
        'Enable OpenGL depth testing.  Disabling also disables writing depth buffer.'
        if enable:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

    def write_depth(self, write):
        'Enable or disable writing to depth buffer.'
        GL.glDepthMask(write)

    def enable_backface_culling(self, enable):
        if enable:
            GL.glEnable(GL.GL_CULL_FACE)
        else:
            GL.glDisable(GL.GL_CULL_FACE)

    def enable_blending(self, enable):
        'Enable OpenGL alpha blending.'
        if enable:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

    def blend_alpha(self, alpha_factor = True):
        'Control whether or not brightness is multiplied by alpha value.'
        src = GL.GL_SRC_ALPHA if alpha_factor else GL.GL_ONE
        GL.glBlendFunc(src, GL.GL_ONE_MINUS_SRC_ALPHA)

    def blend_add(self, f):
        GL.glBlendColor(f, f, f, f)
        GL.glBlendFunc(GL.GL_CONSTANT_COLOR, GL.GL_ONE)
        GL.glEnable(GL.GL_BLEND)

    def blend_max(self, enable):
        # Used for maximum intensity projection texture rendering.
        GL.glBlendEquation(GL.GL_MAX if enable else GL.GL_FUNC_ADD)

    def enable_xor(self, enable):
        if enable:
            GL.glLogicOp(GL.GL_XOR)
            GL.glEnable(GL.GL_COLOR_LOGIC_OP)
        else:
            GL.glDisable(GL.GL_COLOR_LOGIC_OP)

    def flush(self):
        GL.glFlush()

    def finish(self):
        GL.glFinish()

    def draw_front_buffer(self, front):
        GL.glDrawBuffer(GL.GL_FRONT if front else GL.GL_BACK)

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

        if self.rendering_to_screen():
            b = GL.GL_FRONT if front_buffer else GL.GL_BACK
        else:
            b = GL.GL_COLOR_ATTACHMENT0
        GL.glReadBuffer(b)
        GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)

        return rgba

    def set_stereo_buffer(self, eye_num):
        '''
        Set the draw buffer to GL_BACK_LEFT for eye_num = 0
        or GL_BACK_RIGHT for eye_num = 1 or GL_BACK for eye_num = None.
        '''
        self.full_viewport()
        if self.rendering_to_screen():
            if eye_num == 0:
                b = GL.GL_BACK_LEFT
            elif eye_num == 1:
                b = GL.GL_BACK_RIGHT
            elif eye_num is None:
                b = GL.GL_BACK
            GL.glDrawBuffer(b)
            self.current_framebuffer().set_draw_buffer(b)

    def start_depth_render(self, framebuffer, texture_unit, center, radius, size):

        # Set projection matrix to be orthographic and span all models.
        rinv = 1/radius
        from numpy import array, float64
        pm = array(((rinv, 0, 0, 0),
                    (0, rinv, 0, 0),
                    (0, 0, -rinv, 0),
                    (0, 0, -1, 1)), float64)  # orthographic projection along z
        self.set_projection_matrix(pm)

        # Make a framebuffer for depth texture rendering
        fb = framebuffer
        if fb is None or fb.width != size:
            if fb:
                fb.delete()
            dt = Texture()
            dt.initialize_depth((size, size))
            fb = Framebuffer('depth map', self.opengl_context, depth_texture=dt, color=False)
            if not fb.activate():
                fb.delete()
                return None           # Requested size exceeds framebuffer limits

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

    def _texture_window(self, texture, shader_options):
        tw = self._texture_win
        if tw is None:
            self._texture_win = tw = TextureWindow(self)
        tw.activate()
        texture.bind_texture()
        self._opengl_shader(shader_options)
        return tw

    def allow_equal_depth(self, equal):
        GL.glDepthFunc(GL.GL_LEQUAL if equal else GL.GL_LESS)

    def depth_invert(self, invert):
        GL.glDepthFunc(GL.GL_GREATER if invert else GL.GL_LESS)
        GL.glClearDepth(0.0 if invert else 1.0)

    def set_depth_range(self, min, max):
        # # Get z-fighting with screen depth copied to framebuffer object
        # # on Mac/Nvidia
        # GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthRange(min, max)

    def finish_rendering(self):
        GL.glFinish()

    def set_stereo_360_params(self, camera_origin = None, camera_y = None, x_shift = None):
        '''
        Shifts scene vertices to effectively make left/right eye camera positions face the
        vertex being rendered.
        '''
        if camera_origin is None:
            camera_origin, camera_y, x_shift = self._stereo_360_params
        else:
            self._stereo_360_params = (camera_origin, camera_y, x_shift)

        p = self.current_shader_program
        if p is not None and p.capabilities & self.SHADER_STEREO_360:
            p.set_float4("camera_origin_and_shift", tuple(camera_origin) + (x_shift,))
            p.set_float4("camera_vertical", tuple(camera_y) + (0,))

    def _set_depth_texture_shader_variables(self, shader):
        shader.set_integer("tex_depth_2d", self.depth_texture_unit)
        znear, zfar = self._near_far_clip
        shader.set_vector2("tex_depth_projection", (znear, zfar/(zfar-znear)))
        shader.set_vector3("tex_depth_scale", self._depth_texture_scale)

    def set_depth_texture_parameters(self, tex_coord_xscale, tex_coord_yscale, zscale):
        self._depth_texture_scale = (tex_coord_xscale, tex_coord_yscale, zscale)
        p = self.current_shader_program
        if p is not None and p.capabilities & self.SHADER_DEPTH_TEXTURE:
            self._set_depth_texture_shader_variables(p)


class Shadow:
    '''Render a directional shadow.'''

    def __init__(self, render, texture_unit = 1):
        self._render = render

        self._shadow_map_framebuffer = None
        self._shadow_texture_unit = texture_unit
        self._shadow_transform = None        # Map scene coordinates to shadow map texture coordinates.
        self._shadow_view_transform = None   # Map camera coordinates to shadow map texture coordinates.

    def delete(self):
        fb = self._shadow_map_framebuffer
        if fb:
            fb.delete(make_current = True)
            self._shadow_map_framebuffer = None

    def use_shadow_map(self, camera, drawings):
        '''
        Compute shadow map textures for specified drawings.
        Does not include child drawings.
        '''
        r = self._render
        lp = r.lighting
        if not lp.shadows:
            return False

        # Compute light direction in scene coords.
        kl = lp.key_light_direction
        if r.recording_opengl:
            light_direction = lambda c=camera, kl=kl: c.position.transform_vector(kl)
        else:
            light_direction = camera.position.transform_vector(kl)

        # Compute drawing bounds so shadow map can cover all drawings.
        # TODO: Shadow bounds should exclude completely transparent drawings
        #       if transparent do not cast shadows.
        center, radius, sdrawings = _shadow_bounds(drawings)
        if center is None or radius == 0:
            return False

        # Compute shadow map depth texture
        size = lp.shadow_map_size
        self._start_rendering_shadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # Compute light view and scene to shadow map transforms
        bias = lp.shadow_depth_bias
        lvinv, stf = self._shadow_transforms(light_direction, center, radius, bias)
        self._shadow_transform = stf      # Scene to shadow map texture coordinates
        r.set_view_matrix(lvinv)
        from .drawing import draw_depth
        draw_depth(r, sdrawings, opaque_only = not r.material.transparent_cast_shadows)

        shadow_map = self._finish_rendering_shadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(self._shadow_texture_unit)

        return True

    def set_shadow_view(self, camera_position):
        stf = self._shadow_transform * camera_position
        # Transform from camera coordinates to shadow map texture coordinates.
        self._shadow_view_transform = stf
        r = self._render
        p = r.current_shader_program
        if p is not None:
            c = p.capabilities
            if r.SHADER_SHADOW & c and r.SHADER_LIGHTING & c:
                p.set_matrix("shadow_transform", stf.opengl_matrix())

    def _start_rendering_shadowmap(self, center, radius, size=1024):

        r = self._render
        fb = r.start_depth_render(self._shadow_map_framebuffer,
                                  self._shadow_texture_unit,
                                  center, radius, size)
        self._shadow_map_framebuffer = fb

    def _finish_rendering_shadowmap(self):

        r = self._render
        r.draw_depth_only(False)
        fb = r.pop_framebuffer()
        return fb.depth_texture

    def _shadow_transforms(self, light_direction, center, radius, depth_bias=0.005):

        r = self._render
        if r.recording_opengl:
            from . import gllist
            s = gllist.ShadowMatrixFunc(r, light_direction, center, radius, depth_bias)
            return (s.lvinv, s.stf)

        # Compute the view matrix looking along the light direction.
        from chimerax.geometry import normalize_vector, orthonormal_frame, translation, scale
        ld = normalize_vector(light_direction)
        # Light view frame:
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse(is_orthonormal = True)  # Scene to light view coordinates

        # Project orthographic along z to (0, 1) texture coords.
        stf = translation((0.5, 0.5, -depth_bias)) * scale((0.5/radius, 0.5/radius, -0.5/radius)) * lvinv

        fb = r.current_framebuffer()
        w, h = fb.width, fb.height
        if r.current_viewport != (0, 0, w, h):
            # Using a subregion of shadow map to handle multiple shadows in
            # one texture.  Map scene coordinates to subtexture.
            x, y, vw, vh = r.current_viewport
            stf = translation((x / w, y / w, 0)) * scale((vw / w, vh / h, 1)) * stf

        return lvinv, stf

    def _set_shadow_shader_variables(self, shader):
        shader.set_integer("shadow_map", self._shadow_texture_unit)
        stf = self._shadow_view_transform
        if stf is not None:
            shader.set_matrix("shadow_transform", stf.opengl_matrix())

def _shadow_bounds(drawings):
    '''
    Compute bounding box for drawings, not including child drawings.
    '''
    # TODO: This code is incorrectly including child drawings in bounds calculation.
    sdrawings = [d for d in drawings if d.casts_shadows]
    from chimerax.geometry import bounds
    b = bounds.union_bounds(d.bounds() for d in sdrawings
                            if not getattr(d, 'skip_bounds', False))
    center = None if b is None else b.center()
    radius = None if b is None else b.radius()
    return center, radius, sdrawings

class Multishadow:
    '''Render shadows from several directions for ambient occlusion lighting.'''

    def __init__(self, render, texture_unit = 2):
        self._render = render

        # cached state
        self._multishadow_dir = None
        from chimerax.geometry import Places
        self._multishadow_transforms = Places()
        self._multishadow_depth = None
        self._multishadow_current_params = None
        self.multishadow_update_needed = False

        self._multishadow_map_framebuffer = None
        self._multishadow_texture_unit = texture_unit
        self._max_multishadows = None
        self._multishadow_view_transforms = Places() # Includes camera view.

        # near to far clip depth for shadow map, needed to normalize shadow direction vector.
        self._multishadow_depth = None

        # Uniform buffer object for shadow matrices:
        self._multishadow_matrix_buffer_id = None
        self._multishadow_uniform_block = 2     # Uniform block number

    def delete(self):
        fb = self._multishadow_map_framebuffer
        if fb:
            fb.delete(make_current = True)
            self._multishadow_map_framebuffer = None

        mmb = self._multishadow_matrix_buffer_id
        if mmb is not None:
            self._render.make_current()
            GL.glDeleteBuffers(1, [mmb])
            self._multishadow_matrix_buffer_id = None

    def use_multishadow_map(self, drawings):
        r = self._render
        lp = r.lighting
        if lp.multishadow == 0:
            return False
        mat = r.material
        msp = (lp.multishadow, lp.multishadow_map_size, lp.multishadow_depth_bias,
               mat.transparent_cast_shadows, mat.meshes_cast_shadows)
        if self._multishadow_current_params != msp:
            self.multishadow_update_needed = True

        if self.multishadow_update_needed:
            self._multishadow_transforms = []
            self.multishadow_update_needed = False

        light_directions = self._multishadow_directions()
        if len(self._multishadow_transforms) == len(light_directions):
            # Bind shadow map for subsequent rendering of shadows.
            self._bind_depth_texture()
            return True

        # Compute drawing bounds so shadow map can cover all drawings.
        center, radius, sdrawings = _shadow_bounds(drawings)
        if center is None or radius == 0:
            return False

        # Compute shadow map depth texture
        size = lp.multishadow_map_size
        self._start_rendering_multishadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer


        nl = len(light_directions)
        from numpy import empty, float64
        mstf_array = empty((nl,3,4), float64)
        from .drawing import draw_depth
        from math import ceil, sqrt
        d = int(ceil(sqrt(nl)))     # Number of subtextures along each axis
        s = size // d               # Subtexture size.
        bias = lp.multishadow_depth_bias

        for l in range(nl):
            x, y = (l % d), (l // d)
            r.set_viewport(x * s, y * s, s, s)
            lvinv, tf = r.shadow._shadow_transforms(light_directions[l], center, radius, bias)
            r.set_view_matrix(lvinv)
            mstf_array[l,:,:] = tf.matrix
            draw_depth(r, sdrawings, opaque_only = not mat.transparent_cast_shadows)
        from chimerax.geometry import Places
        mstf = Places(place_array = mstf_array)

        self._finish_rendering_multishadowmap()

        # Bind shadow map for subsequent rendering of shadows.
        self._bind_depth_texture()

        # TODO: Clear shadow cache whenever scene changes
        self._multishadow_current_params = msp
        self._multishadow_transforms = mstf
        self._multishadow_depth = msd = 2 * radius
#        self.set_multishadow_transforms(mstf, None, msd)

        return True

    def set_multishadow_view(self, camera_position):
        '''
        Although shadows are independent of view direction, the fragment shader
        uses camera coordinates.
        '''
        ctf = camera_position
        stf = self._multishadow_transforms
        vtf = self._multishadow_view_transforms
        r = self._render
        # Transform from camera coordinates to shadow map texture coordinates.
        if r.recording_opengl:
            from .gllist import Mat34Func
            self._multishadow_view_transforms = Mat34Func('multishadow matrices', lambda: (stf * ctf()), len(stf))
        else:
            from chimerax.geometry import multiply_transforms
            multiply_transforms(stf, ctf, result = vtf)

        # TODO: Issue warning if maximum number of shadows exceeded.
        maxs = self.max_multishadows()

        mm = vtf.opengl_matrices()
        if not r.recording_opengl:
            mm = mm[:maxs, :, :]
        offset = 0
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._multishadow_matrix_buffer())
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, offset, mm.nbytes, mm)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

        p = r.current_shader_program
        if p is not None:
            c = p.capabilities
            if r.SHADER_MULTISHADOW & c and r.SHADER_LIGHTING & c:
                self._set_multishadow_shader_variables(p)

    def max_multishadows(self):
        'Maximum number of shadows to cast.'
        m = self._max_multishadows
        if m is None:
            m = GL.glGetIntegerv(GL.GL_MAX_UNIFORM_BLOCK_SIZE)      # OpenGL requires >= 16384.
            if m > 2 ** 17:                                         # Limit maximum to limit size
                m = 2 ** 17                                         #    of buffer in shader.
            m = m // 64                                             # 64 bytes per matrix.
            self._max_multishadows = m
        return m

    def _start_rendering_multishadowmap(self, center, radius, size=1024):
        r = self._render
        fb = r.start_depth_render(self._multishadow_map_framebuffer,
                                  self._multishadow_texture_unit,
                                  center, radius, size)
        self._multishadow_map_framebuffer = fb

    def _finish_rendering_multishadowmap(self):
        r = self._render
        return r.shadow._finish_rendering_shadowmap()

    def _multishadow_directions(self):
        directions = self._multishadow_dir
        n = self._render.lighting.multishadow
        if directions is None or len(directions) != n:
            from chimerax.geometry import sphere
            self._multishadow_dir = directions = sphere.sphere_points(n)
        return directions

    def _bind_depth_texture(self):
        dt = self._multishadow_map_framebuffer.depth_texture
        dt.bind_texture(self._multishadow_texture_unit)

    def _set_multishadow_shader_constants(self, shader):
        # Set the multishadow texture unit and the matrix uniform block unit for the shader.
        shader.set_integer("multishadow_map", self._multishadow_texture_unit)
        pid = shader.program_id
        bi = GL.glGetUniformBlockIndex(pid, b'shadow_matrix_block')
        bslot = self._multishadow_uniform_block
        GL.glUniformBlockBinding(pid, bi, bslot)

    def _multishadow_matrix_buffer(self):
        b = self._multishadow_matrix_buffer_id
        if b is None:
            # Create uniform buffer object for shadow matrices.
            self._multishadow_matrix_buffer_id = b = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, b)
            nbytes = 64 * self.max_multishadows()
            GL.glBufferData(GL.GL_UNIFORM_BUFFER, nbytes, pyopengl_null(), GL.GL_DYNAMIC_DRAW)
            bslot = self._multishadow_uniform_block
            GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, bslot, b)
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)
        return b

    def _set_multishadow_shader_variables(self, shader):
        m = self._multishadow_view_transforms
        if m is None:
            return

        maxs = self.max_multishadows()
        shader.set_integer("shadow_count", min(maxs, len(m)))
        if shader.capabilities & Render.SHADER_LIGHTING_NORMALS:
            shader.set_float("shadow_depth", self._multishadow_depth)

class Offscreen:
    '''Offscreen framebuffer for 16-bit color depth.'''

    def __init__(self):
        self.enabled = False
        self._offscreen_framebuf = None

    def delete(self):
        fb = self._offscreen_framebuf
        if fb:
            fb.delete(make_current = True)
            self._offscreen_framebuf = None

    def start(self, render):
        # TODO: Should only push new framebuffer if current framebuffer is not
        #   an offscreen buffer.
        r = render
        fb = self._offscreen_framebuffer(r)
        r.push_framebuffer(fb)

    def finish(self, render):
        r = render
        fb = r.pop_framebuffer()
        cfb = r.current_framebuffer()
        cfb.copy_from_framebuffer(fb, depth=False)

    def _offscreen_framebuffer(self, render):
        r = render
        w,h = r.render_size()
        ofb = self._offscreen_framebuf
        if ofb and (ofb.width != w or ofb.height != h):
            ofb.delete()
            ofb = None
        if ofb is None:
            ofb = Framebuffer('offscreen', r.opengl_context, w, h)
            self._offscreen_framebuf = ofb
        return ofb

class Silhouette:
    '''Draw silhouette edges.'''

    def __init__(self):
        self.enabled = False
        self.thickness = 1           # pixels
        self.color = (0, 0, 0, 1)    # black
        self.depth_jump = 0.03       # fraction of scene depth
        self.perspective_near_far_ratio = 1 # Needed for handling depth buffer scaling
        self._silhouette_framebuf = None

    def delete(self):
        fb = self._silhouette_framebuf
        if fb:
            fb.delete(make_current = True)
            self._silhouette_framebuf = None

    def start_silhouette_drawing(self, render):
        r = render
        fb = self._silhouette_framebuffer(r)
        r.push_framebuffer(fb)

    def finish_silhouette_drawing(self, render):
        r = render
        fb = r.pop_framebuffer()
        cfb = r.current_framebuffer()
        cfb.copy_from_framebuffer(fb, depth=False)
        self._draw_depth_outline(render, fb.depth_texture, self.thickness,
                                 self.color, self.depth_jump,
                                 self.perspective_near_far_ratio)

    def draw_silhouette(self, render):
        r = render
        fb = r.current_framebuffer()

        # Can't have depth texture attached to framebuffer and sampled.
        fb.attach_depth_texture(None)  # Detach depth texture

        self._draw_depth_outline(render, fb.depth_texture, self.thickness,
                                 self.color, self.depth_jump,
                                 self.perspective_near_far_ratio)

        fb.attach_depth_texture(fb.depth_texture)  # Reattach depth texture

    def _silhouette_framebuffer(self, render):
        r = render
        size = r.render_size()
        alpha = r.current_framebuffer().alpha
        sfb = self._silhouette_framebuf
        if sfb and (size[0] != sfb.width or size[1] != sfb.height or alpha != sfb.alpha):
            sfb.delete()
            sfb = None
        if sfb is None:
            dt = Texture()
            dt.linear_interpolation = False
            dt.initialize_depth(size, depth_compare_mode=False)
            sfb = Framebuffer('silhouette', r.opengl_context, depth_texture=dt, alpha=alpha)
            self._silhouette_framebuf = sfb
        return sfb

    def _draw_depth_outline(self, render, depth_texture, thickness=1,
                            color=(0, 0, 0, 1), depth_jump=0.03,
                            perspective_near_far_ratio=1):
        # Render pixels with depth in depth_texture less than neighbor pixel
        # by at least depth_jump. The depth buffer is not used.
        r = render
        tc = r._texture_window(depth_texture, r.SHADER_DEPTH_OUTLINE)

        p = r.current_shader_program
        p.set_rgba("color", color)
        p.set_vector2("jump", (depth_jump, perspective_near_far_ratio))
        w, h = depth_texture.size
        dx, dy = 1.0 / w, 1.0 / h
        p.set_vector3("step", (dx, dy, thickness))
        tc.draw(blend = True)

class Outline:
    '''Draw highlight outlines.'''

    def __init__(self, render):
        self._render = render
        self._mask_framebuf = None

        # Copying depth from default framebuffer does not work for Radeon Pro Vega 20
        # graphics on macOS 10.14.5.  This flag is to work around that bug.
        self.offscreen_outline_needed = False

    def delete(self):
        fb = self._mask_framebuf
        if fb:
            fb.delete(make_current = True)
            self._mask_framebuf = None

    def set_outline_mask(self):
        '''Copy framebuffer depth to outline framebuffer.  Only highlighted
        objects at equal depth or in front will be outlined by start_rendering_outline().
        This routine must be called before start_rendering_outline().
        '''
        r = self._render
        mfb = self._mask_framebuffer()
        cfb = r.current_framebuffer()
        cfb.copy_to_framebuffer(mfb, color=False)

    def start_rendering_outline(self):
        '''Must call set_outline_depth() before invoking this routine.'''
        r = self._render
        fb = r.current_framebuffer()
        mfb = self._mask_framebuffer()
        r.push_framebuffer(mfb)
        last_bg = r._last_background_color
        r.set_background_color((0, 0, 0, 0))
        r.draw_background(depth = False)
        r.set_background_color(last_bg)
        # Use unlit all white color for drawing mask.
        # Outline code requires non-zero red component.
        r.disable_shader_capabilities(r.SHADER_VERTEX_COLORS
                                      | r.SHADER_TEXTURE_2D
                                      | r.SHADER_TEXTURE_3D
                                      | r.SHADER_TEXTURE_CUBEMAP
                                      | r.SHADER_LIGHTING)
        r.enable_capabilities |= r.SHADER_ALL_WHITE
        # Depth test GL_LEQUAL results in z-fighting:
        r.set_depth_range(0, 0.9995)

    def finish_rendering_outline(self, color=(0,1,0,1), pixel_width=1):
        r = self._render
        r.pop_framebuffer()
        r.disable_shader_capabilities(0)
        r.enable_capabilities &= ~r.SHADER_ALL_WHITE
        r.set_depth_range(0, 1)
        t = self._mask_framebuf.color_texture
        self._draw_texture_mask_outline(t, color=color, pixel_width=pixel_width)

    def _draw_texture_mask_outline(self, texture, color=(0, 1, 0, 1), pixel_width=1):

        # Render outline of region where texture red > 0.
        # Outline pixels have red = 0 in texture mask but are adjacent
        # in one of 4 directions to pixels with red > 0.
        # The depth buffer is not used.  (Depth buffer was used to handle
        # occlusion in the mask texture passed to this routine.)
        w, h = texture.size
        dx, dy = pixel_width / w, pixel_width / h
        r = self._render
        tc = r._texture_window(texture, r.SHADER_TEXTURE_OUTLINE)
        p = r.current_shader_program
        p.set_vector2('step', (dx, dy))
        p.set_rgba('color', color)
        tc.draw()

    def _mask_framebuffer(self):
        r = self._render
        size = r.render_size()
        mfb = self._mask_framebuf
        w, h = size
        if mfb and mfb.width == w and mfb.height == h:
            return mfb
        if mfb:
            mfb.delete()
        t = Texture()
        t.linear_interpolation = False
        t.initialize_8_bit(size)
        self._mask_framebuf = mfb = Framebuffer('mask', r.opengl_context, color_texture=t)
        return mfb

class BlendTextures:
    '''
    Blend 2D textures adding RGB values and combining opacities.
    '''
    def __init__(self, render):
        self._render = render
        self._blend_framebuffer = None

    def delete(self):
        fb = self._blend_framebuffer
        if fb:
            fb.delete(make_current = True)
            self._blend_framebuffer = None

    def start_blending(self, texture):
        r = self._render
        fb = self._blending_framebuffer(texture)
        r.push_framebuffer(fb)
        r.set_background_color((0,0,0,0))
        if texture.dimension == 3:
            self._background_cleared = False
        else:
            r.draw_background()
            self._background_cleared = True
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_ONE, GL.GL_ONE, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

    def blend(self, texture, modulation_color, colormap = None, colormap_range = None):
        # TODO: Should use nearest pixel texture lookup for speed.
        r = self._render
        if colormap is None:
            tw = r._texture_window(texture, r.SHADER_BLEND_TEXTURE_2D)
        else:
            tw = r._texture_window(texture, r.SHADER_BLEND_TEXTURE_2D | r.SHADER_BLEND_COLORMAP)
            r.set_colormap(colormap, colormap_range, texture)
        r.set_model_color(modulation_color)
        tw.draw()

    def blend3d(self, texture, modulation_color, dest_tex, colormap = None, colormap_range = None):
        r = self._render
        fb = r.current_framebuffer()
        if colormap is None:
            tw = r._texture_window(texture, r.SHADER_BLEND_TEXTURE_3D)
        else:
            tw = r._texture_window(texture, r.SHADER_BLEND_TEXTURE_3D | r.SHADER_BLEND_COLORMAP)
            r.set_colormap(colormap, colormap_range, texture)
        p = r.current_shader_program
        r.set_model_color(modulation_color)
        linear = texture.linear_interpolation
        texture.set_linear_interpolation(False)  # Use nearest pixel texture lookup for speed.
        n = texture.size[2]
        for k in range(n):
            fb.set_color_buffer(dest_tex, plane = k)
            p.set_float('tex_coord_z', (k+.5)/n)
            if not self._background_cleared:
                r.draw_background()
            tw.draw()
        self._background_cleared = True
        texture.set_linear_interpolation(linear) # Restore linear interpolation mode.

    def finish_blending(self):
        r = self._render
        r.pop_framebuffer()
        r.blend_alpha() # Reset to alpha blending

    def _blending_framebuffer(self, color_texture):
        fb = self._blend_framebuffer
        if fb is None:
            r = self._render
            fb = Framebuffer('blend', r.opengl_context, color_texture=color_texture, depth=False)
            self._blend_framebuffer = fb
        else:
            fb.set_color_buffer(color_texture)
        return fb

# Options used with Render.shader()
shader_options = (
    'SHADER_LIGHTING',
    'SHADER_LIGHTING_NORMALS',
    'SHADER_DEPTH_CUE',
    'SHADER_NO_DEPTH_CUE',
    'SHADER_TEXTURE_2D',
    'SHADER_TEXTURE_3D',
    'SHADER_COLORMAP',
    'SHADER_DEPTH_TEXTURE',
    'SHADER_TEXTURE_CUBEMAP',
    'SHADER_TEXTURE_3D_AMBIENT',
    'SHADER_BLEND_TEXTURE_2D',
    'SHADER_BLEND_TEXTURE_3D',
    'SHADER_BLEND_COLORMAP',
    'SHADER_SHADOW',
    'SHADER_NO_SHADOW',
    'SHADER_MULTISHADOW',
    'SHADER_NO_MULTISHADOW',
    'SHADER_SHIFT_AND_SCALE',
    'SHADER_INSTANCING',
    'SHADER_TEXTURE_OUTLINE',
    'SHADER_DEPTH_OUTLINE',
    'SHADER_VERTEX_COLORS',
    'SHADER_FRAME_NUMBER',
    'SHADER_TRANSPARENT_ONLY',
    'SHADER_OPAQUE_ONLY',
    'SHADER_STEREO_360',
    'SHADER_CLIP_PLANES',
    'SHADER_NO_CLIP_PLANES',
    'SHADER_ALPHA_DEPTH',
    'SHADER_ALL_WHITE',
)
for i, sopt in enumerate(shader_options):
    setattr(Render, sopt, 1 << i)

Render.SHADER_NO_PROJECTION_MATRIX = (Render.SHADER_TEXTURE_OUTLINE |
                                      Render.SHADER_DEPTH_OUTLINE |
                                      Render.SHADER_BLEND_TEXTURE_2D |
                                      Render.SHADER_BLEND_TEXTURE_3D)

def shader_capability_names(capabilities_bit_mask):
    return [name for i, name in enumerate(shader_options)
            if capabilities_bit_mask & (1 << i)]


class Framebuffer:
    '''
    OpenGL framebuffer for off-screen rendering.  Allows rendering colors
    and/or depth to a texture.
    '''
    def __init__(self, name, opengl_context,
                 width=None, height=None,
                 color=True, color_texture=None,
                 depth=True, depth_texture=None,
                 alpha=False):

        self.name = name # For debugging

        if width is not None and height is not None:
            w, h = width, height
        elif color_texture is not None:
            w, h = color_texture.size[:2]
        elif depth_texture is not None:
            w, h = depth_texture.size
        else:
            w = h = None

        self.width = w
        self.height = h
        self.alpha = alpha
        self.viewport = None if w is None else (0, 0, w, h)
        self.color = color
        self.color_texture = color_texture
        self.depth = depth
        self.depth_texture = depth_texture

        self._color_rb = None
        self._color_bits = opengl_context._framebuffer_color_bits # 8 or 16-bit depth
        self._depth_rb = None
        self._draw_buffer = GL.GL_COLOR_ATTACHMENT0
        self._deleted = False

        self._fbo = None
        if w is None:
            self._fbo = 0 # Default framebuffer
            self._draw_buffer = opengl_context.default_draw_target

        self._opengl_context = opengl_context

    def _create_framebuffer(self):
        w, h = self.width, self.height
        if w is None:
            fbo = 0
        elif not self.valid_size(w, h):
            raise OpenGLError('Frame buffer invalid size %d, %d' % (w,h))
        else:
            if self._color_rb is None and self.color and self.color_texture is None and w is not None:
                self._color_rb = self.color_renderbuffer(w, h, self.alpha)
            if self._depth_rb is None and self.depth and self.depth_texture is None and w is not None:
                self._depth_rb = self.depth_renderbuffer(w, h)
            fbo = self._create_fbo(self.color_texture or self._color_rb,
                                   self.depth_texture or self._depth_rb)
        return fbo

    def _create_fbo(self, color_buf, depth_buf):
        fbo = GL.glGenFramebuffers(1)
        self._opengl_context._framebuffers.add(self)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        if isinstance(color_buf, Texture):
            level = 0
            if color_buf.dimension == 2:
                target = GL.GL_TEXTURE_2D if not color_buf.is_cubemap else GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                          GL.GL_COLOR_ATTACHMENT0,
                                          target, color_buf.id, level)
            elif color_buf.dimension == 3:
                # Use first plane of 3d texture as color buffer.
                # Change plane with set_color_buffer().
                GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER,
                                             GL.GL_COLOR_ATTACHMENT0,
                                             color_buf.id, level, 0)
        elif color_buf is not None:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                         GL.GL_COLOR_ATTACHMENT0,
                                         GL.GL_RENDERBUFFER, color_buf)
        else:
            # Need this or glCheckFramebufferStatus() fails with no color buffer.
            GL.glDrawBuffer(GL.GL_NONE)
            GL.glReadBuffer(GL.GL_NONE)
            self._draw_buffer = GL.GL_NONE

        if isinstance(depth_buf, Texture):
            self.attach_depth_texture(depth_buf)
        elif depth_buf is not None:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                         GL.GL_DEPTH_ATTACHMENT,
                                         GL.GL_RENDERBUFFER, depth_buf)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self._fbo = fbo # Need to set attribute for delete to release attachments.
            self.delete()
            # TODO: Need to rebind previous framebuffer.
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            raise OpenGLError('Framebuffer %s creation failed, status %d, color buffer %s, depth buffer %s'
                              % (self.name, status, color_buf, depth_buf))

        return fbo

    def set_color_buffer(self, color_texture, plane = 0):
        '''
        Has side effect that framebuffer is bound.
        '''
        level = 0
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer_id)
        if color_texture.dimension == 2:
            target = GL.GL_TEXTURE_2D if not color_texture.is_cubemap else GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      target, color_texture.id, level)
        elif color_texture.dimension == 3:
            GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                         color_texture.id, level, plane)
        w,h = color_texture.size[:2]
        self.width = w
        self.height = h
        self.viewport = (0,0,w,h)

    def attach_depth_texture(self, depth_texture):
        tid = 0 if depth_texture is None else depth_texture.id
        level = 0
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D,
                                  tid, level)

    def __del__(self):
        if not self._deleted and self._fbo != 0:
            raise OpenGLError('OpenGL framebuffer "%s" was not deleted before graphics.Framebuffer destroyed'
                               % self.name)

    def delete(self, make_current = False):
        self._deleted = True

        if make_current:
            self._opengl_context.make_current()

        self._release()

        ct = self.color_texture
        if ct is not None:
            ct.delete_texture()
            self.color_texture = None

        dt = self.depth_texture
        if dt is not None:
            dt.delete_texture()
            self.depth_texture = None

    def _release(self):
        '''Delete opengl framebuffer allowing it to be recreated.'''
        fbo = self._fbo
        if fbo is not None and fbo != 0:
            if self._color_rb is not None:
                GL.glDeleteRenderbuffers(1, (self._color_rb,))
            if self._depth_rb is not None:
                GL.glDeleteRenderbuffers(1, (self._depth_rb,))
            GL.glDeleteFramebuffers(1, (fbo,))
            self._color_rb = self._depth_rb = None
            self._fbo = None
            self._opengl_context._framebuffers.discard(self)

    def valid_size(self, width, height):

        max_rb_size = GL.glGetInteger(GL.GL_MAX_RENDERBUFFER_SIZE)
        max_tex_size = GL.glGetInteger(GL.GL_MAX_TEXTURE_SIZE)
        max_size = min(max_rb_size, max_tex_size)
        return width <= max_size and height <= max_size

    def set_color_bits(self, bits):

        if bits == self._color_bits:
            return

        self._color_bits = bits
        if self._color_rb is not None:
            GL.glDeleteRenderbuffers(1, (self._color_rb,))
            self._color_rb = self.color_renderbuffer(self.width, self.height, self.alpha)

    def color_renderbuffer(self, width, height, alpha = False):

        color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color_rb)
        if self._color_bits == 8:
            fmt = GL.GL_RGBA8 if alpha else GL.GL_RGB8
        elif self._color_bits == 16:
            fmt = GL.GL_RGBA16 if alpha else GL.GL_RGB16
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, width, height)
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

    def set_cubemap_face(self, face):
        level = 0
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X+face,
                                  self.color_texture.id, level)

    @property
    def framebuffer_id(self):
        fbo = self._fbo
        if fbo is None:
            self._fbo = fbo = self._create_framebuffer()
        return fbo

    def activate(self):
        fbo = self.framebuffer_id
        if fbo is None:
            return False
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glDrawBuffer(self._draw_buffer)
        return True

    def copy_from_framebuffer(self, framebuffer, color=True, depth=True):
        # Copy current framebuffer contents to another framebuffer.  This
        # leaves read and draw framebuffers set to the current framebuffer.
        from_id = framebuffer.framebuffer_id
        to_id = self.framebuffer_id
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, from_id)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, to_id)
        what = GL.GL_COLOR_BUFFER_BIT if color else 0
        if depth:
            what |= GL.GL_DEPTH_BUFFER_BIT
        w, h = framebuffer.width, framebuffer.height
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, what, GL.GL_NEAREST)
        # Restore read buffer
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, to_id)

    def copy_to_framebuffer(self, framebuffer, color=True, depth=True):
        # Copy current framebuffer contents to another framebuffer.  This
        # leaves read and draw framebuffers set to the current framebuffer.
        from_id = self.framebuffer_id
        to_id = framebuffer.framebuffer_id
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, from_id)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, to_id)
        what = GL.GL_COLOR_BUFFER_BIT if color else 0
        if depth:
            what |= GL.GL_DEPTH_BUFFER_BIT
        w, h = framebuffer.width, framebuffer.height
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, what, GL.GL_NEAREST)
        # Restore draw buffer
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, from_id)

    def set_draw_buffer(self, buffer_name):
        # When framebuffer is activated, glDrawBuffer() is set using this value.
        self._draw_buffer = buffer_name

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

    def set_default_parameters(self, background_color = None):
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

        self.depth_cue = True
        "Is depth cuing enabled."

        self.depth_cue_start = 0.5
        "Fraction of distance from near to far clip plane where dimming starts."

        self.depth_cue_end = 1.0
        "Fraction of distance from near to far clip plane where dimming ends."

        self.depth_cue_color = (0, 0, 0) if background_color is None else tuple(background_color[:3])
        "Color to fade towards."

        self.move_lights_with_camera = True
        "Whether lights are attached to camera, or fixed in the scene."

        self.shadows = False
        "Does key light cast shadows."

        self.shadow_map_size = 2048
        "Size of 2D opengl texture used for casting shadows."

        self.shadow_depth_bias = 0.005
        "Offset as fraction of scene depth for avoiding surface self-shadowing."

        self.multishadow = 0
        '''
        The number of shadows to use for ambient shadowing,
        for example, 64 or 128.  To turn off ambient shadows specify 0
        shadows.  Shadows are cast from uniformly distributed directions.
        This is GPU intensive, each shadow requiring a texture lookup.
        '''

        self.multishadow_map_size = 1024
        '''Size of 2D opengl texture used for casting ambient shadows.
        This texture is tiled to hold shadow maps for all directions.'''

        self.multishadow_depth_bias = 0.01
        "Offset as fraction of scene depth for avoiding surface ambient self-shadowing."

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

        self.transparent_cast_shadows = False
        "Do transparent objects cast shadows."

        self.meshes_cast_shadows = False
        "Do mesh style Drawings cast shadows."

class Bindings:
    '''
    Use an OpenGL vertex array object to save buffer bindings.
    The bindings are for a specific shader program since they use the
    shader variable ids.
    '''
    attribute_id = {'position': 0, 'tex_coord': 1, 'normal': 2, 'vcolor': 3,
                    'instance_shift_and_scale': 4, 'instance_placement': 5}

    def __init__(self, name, opengl_context):
        self._name = name # Used for debugging
        self._vao_id = None
        self._bound_attr_ids = {}        # Maps buffer to list of ids
        self._bound_attr_buffers = {} # Maps attribute id to bound buffer (or None).
        self._bound_element_buffer = None
        self._opengl_context = opengl_context

    def __del__(self):
        if self._vao_id is not None:
            raise OpenGLError('OpenGL vertex array object was not deleted before graphics.Bindings destroyed')

    def delete_bindings(self):
        'Delete the OpenGL vertex array object.'
        self._release()

    def _release(self):
        'Delete the OpenGL vertex array object allowing it to be recreated.'
        if self._vao_id is not None:
            GL.glDeleteVertexArrays(1, (self._vao_id,))
            self._vao_id = None
            self._opengl_context._bindings.discard(self)

    def activate(self):
        'Activate the bindings by binding the OpenGL vertex array object.'
        if self._vao_id is None:
            self._vao_id = GL.glGenVertexArrays(1)
            self._opengl_context._bindings.add(self)
            GL.glBindVertexArray(self._vao_id)
            for buffer in tuple(self._bound_attr_buffers.values()):
                if buffer is not None:
                    self.bind_shader_variable(buffer)
            eb = self._bound_element_buffer
            if eb:
                self.bind_shader_variable(eb)
        GL.glBindVertexArray(self._vao_id)

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
            for a in self._bound_attr_ids.get(buffer, []):
                if self._bound_attr_buffers[a] is buffer:
                    GL.glDisableVertexAttribArray(a)
                    self._bound_attr_buffers[a] = None
            self._bound_attr_ids[buffer] = []
            if btype == GL.GL_ELEMENT_ARRAY_BUFFER:
                GL.glBindBuffer(btype, 0)
            return

        vname = buffer.shader_variable_name
        if vname is None:
            if btype == GL.GL_ELEMENT_ARRAY_BUFFER:
                # Element array buffer binding is saved in VAO.
                GL.glBindBuffer(btype, buf_id)
                self._bound_element_buffer = buffer
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
            self._bound_attr_ids[buffer] = [attr_id]
            self._bound_attr_buffers[attr_id] = buffer
        else:
            # Matrices use multiple vector attributes
            esize = buffer.array_element_bytes()
            abytes = ncomp * esize
            stride = nattr * abytes
            bab = self._bound_attr_buffers
            import ctypes
            for a in range(nattr):
                # Pointer arg must be void_p, not an integer.
                p = ctypes.c_void_p(a * abytes)
                a_id = attr_id + a
                GL.glVertexAttribPointer(a_id, ncomp, gtype, normalize, stride, p)
                GL.glEnableVertexAttribArray(a_id)
                GL.glVertexAttribDivisor(a_id, 1 if buffer.instance_buffer else 0)
                bab[a_id] = buffer
            self._bound_attr_ids[buffer] = [attr_id + a for a in range(nattr)]
        GL.glBindBuffer(btype, 0)

        # print('bound shader variable', vname, attr_id, nattr, ncomp)
        return attr_id


def deactivate_bindings():
    GL.glBindVertexArray(0)

from numpy import uint8, uint32, float32


GL_ARRAY_BUFFER = 34962
class BufferType:
    '''
    Describes a shader variable and the vertex buffer object value type
    required and what rendering capabilities are required to use this
    shader variable.
    '''
    def __init__(self, shader_variable_name, buffer_type=GL_ARRAY_BUFFER,
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
TEXTURE_COORDS_BUFFER = BufferType(
    'tex_coord',
    requires_capabilities = (Render.SHADER_TEXTURE_2D |
                             Render.SHADER_TEXTURE_3D |
                             Render.SHADER_TEXTURE_OUTLINE |
                             Render.SHADER_DEPTH_OUTLINE))
GL_ELEMENT_ARRAY_BUFFER = 34963
ELEMENT_BUFFER = BufferType(None, buffer_type=GL_ELEMENT_ARRAY_BUFFER,
                            value_type=uint32)

GL_TRIANGLES = 4
GL_LINES = 1
GL_POINTS = 0
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

        self._deleted_buffer = False

    def __del__(self):
        if not self._deleted_buffer:
            raise OpenGLError('OpenGL buffer "%s" was not deleted before graphics.Buffer destroyed'
                               % self.shader_variable_name)

    def delete_buffer(self):
        'Delete the OpenGL buffer object.'
        self._deleted_buffer = True
        self.release_buffer()

    def release_buffer(self):
        '''
        Releases OpenGL buffer and array data, but Buffer can still be used
        by calling update_buffer_data() to recreate it.
        '''
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

    def size(self):
        barray = self.buffered_array
        return 0 if barray is None else barray.size

    def update_buffer_data(self, data):
        '''
        Update the buffer with data supplied by a numpy array and bind it to
        the associated shader variable.  Return true if the buffer is deleted and replaced.
        '''
        if self._deleted_buffer:
            raise RuntimeError('Attempt to update a deleted buffer')

        bdata = self.buffered_data
        replace_buffer = (data is None or bdata is None
                          or data.shape != bdata.shape)
        if replace_buffer:
            self.release_buffer()

        if data is not None:
            b = GL.glGenBuffers(1) if replace_buffer else self.opengl_buffer
            btype = self.buffer_type
            GL.glBindBuffer(btype, b)
            if data.dtype == self.value_type and data.flags['C_CONTIGUOUS']:
                # PyOpenGL 3.1.5 leaks memory if data not contiguous, PyOpenGL github issue #47.
                d = data
            else:
                d = data.astype(self.value_type, order = 'C')
            size = d.size * d.itemsize        # Bytes
            if replace_buffer:
                GL.glBufferData(btype, size, d, GL.GL_STATIC_DRAW)
            else:
                GL.glBufferSubData(btype, 0, size, d)
            GL.glBindBuffer(btype, 0)
            self.opengl_buffer = b
            self.buffered_array = d
            self.buffered_data = data

        return replace_buffer

    # Element types for Buffer draw_elements()
    triangles = GL_TRIANGLES
    lines = GL_LINES
    points = GL_POINTS

    def draw_elements(self, element_type=triangles, ninst=None, count=None, offset=None):
        '''
        Draw primitives using this buffer as the element buffer.
        All the required buffers are assumed to be already bound using a
        vertex array object including the element buffer.
        '''
        ne = self.buffered_array.size if count is None else count
        if offset is None:
            eo = None
        else:
            import ctypes
            eo = ctypes.c_void_p(offset * self.array_element_bytes())
        if ninst is None:
            GL.glDrawElements(element_type, ne, GL.GL_UNSIGNED_INT, eo)
        else:
            GL.glDrawElementsInstanced(element_type, ne, GL.GL_UNSIGNED_INT, eo, ninst)

    def shader_has_required_capabilities(self, shader):
        if not self.requires_capabilities:
            return True
        # Require at least one capability of list
        return self.requires_capabilities | shader.capabilities


class Shader:
    '''OpenGL shader program with specified capabilities.'''

    def __init__(self, capabilities = 0, max_shadows = 0,
                 vertex_shader_path = None, fragment_shader_path = None):

        self.capabilities = capabilities

        if vertex_shader_path is None:
            from os.path import dirname, join
            vertex_shader_path = join(dirname(__file__), 'vertexShader.txt')

        if fragment_shader_path is None:
            from os.path import dirname, join
            fragment_shader_path = join(dirname(__file__), 'fragmentShader.txt')

        self.program_id = self.compile_shader(vertex_shader_path, fragment_shader_path,
                                              capabilities, max_shadows)
        self.uniform_ids = {}
        self._validated = False # Don't validate program until uniforms set.

    def __str__(self):
        caps = ', '.join(n[7:] for n in shader_capability_names(self.capabilities))
        return 'shader %d, capabilities %s' % (self.program_id, caps)

    def set_integer(self, name, value):
        GL.glUniform1i(self.uniform_id(name), value)

    def set_float(self, name, value):
        GL.glUniform1f(self.uniform_id(name), value)

    def set_vector(self, name, vector):
        GL.glUniform3fv(self.uniform_id(name), 1, vector)

    def set_vector2(self, name, vector):
        GL.glUniform2fv(self.uniform_id(name), 1, vector)

    def set_vector3(self, name, vector):
        GL.glUniform3fv(self.uniform_id(name), 1, vector)

    def set_rgba(self, name, color):
        GL.glUniform4fv(self.uniform_id(name), 1, color)

    def set_float4(self, name, v4, count = 1):
        GL.glUniform4fv(self.uniform_id(name), count, v4)

    def set_matrix(self, name, matrix):
        GL.glUniformMatrix4fv(self.uniform_id(name), 1, False, matrix)

    def uniform_id(self, name):
        uids = self.uniform_ids
        if name in uids:
            uid = uids[name]
        else:
            p = self.program_id
            uid = GL.glGetUniformLocation(p, name.encode('utf-8'))
            if uid == -1:
                raise OpenGLError('Shader does not have uniform variable "%s"\n shader capabilities %s'
                                   % (name, ', '.join(shader_capability_names(self.capabilities))))
            uids[name] = uid
        return uid

    def compile_shader(self, vertex_shader_path, fragment_shader_path,
                       capabilities = 0, max_shadows = 0):

        f = open(vertex_shader_path, 'r')
        vshader = self.insert_define_macros(f.read(), capabilities, max_shadows)
        f.close()

        f = open(fragment_shader_path, 'r')
        fshader = self.insert_define_macros(f.read(), capabilities, max_shadows)
        f.close()

        from OpenGL.GL import shaders
        try:
            vs = shaders.compileShader(vshader, GL.GL_VERTEX_SHADER)
            fs = shaders.compileShader(fshader, GL.GL_FRAGMENT_SHADER)
        except Exception as e:
            raise OpenGLError(str(e))

        prog_id = self.compile_program(vs, fs)

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

    def compile_program(self, vs, fs):
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)
        GL.glLinkProgram(program)
        link_status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
        if link_status == GL.GL_FALSE:
            raise OpenGLError( 'Link failure (%s): %s'
                                % (link_status, GL.glGetProgramInfoLog(program)))
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return program

    def validate_program(self):
        # Only validate after setting Sampler uniforms.  Validation finds collisions
        # using the same texture unit for multiple samplers.  Usually only validate in debug builds.
        if self._validated:
            return
        self._validated = True
        p = self.program_id
        GL.glValidateProgram(p)
        validation = GL.glGetProgramiv(p, GL.GL_VALIDATE_STATUS )
        if validation == GL.GL_FALSE:
            raise OpenGLError('OpenGL Program validation failure (%r): %s'
                               % (validation, GL.glGetProgramInfoLog(p)))

    # Add #define lines after #version line of shader
    def insert_define_macros(self, shader, capabilities, max_shadows):
        '''Private. Puts "#define" statements in shader program templates
        to specify shader capabilities.'''
        deflines = ['#define %s 1' % sopt.replace('SHADER_', 'USE_')
                    for sopt in shader_capability_names(capabilities)]
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
    The OpenGL texture is only created when the bind_texture() method
    is called.  A reference to the array data is held until the OpenGL
    texture is created.
    '''
    def __init__(self, data=None, dimension=2, cube_map=False,
                 linear_interpolation=True, clamp_to_edge=False):

        # PyOpenGL 3.1.5 leaks memory if data not contiguous, PyOpenGL github issue #47.
        d = data if data is None or data.flags['C_CONTIGUOUS'] else data.copy()
        self.data = d
        self.id = None
        self.dimension = dimension
        self.size = None
        self._array_shape = None
        self._numpy_dtype = None
        self.linear_interpolation = linear_interpolation
        self.is_cubemap = cube_map
        self.clamp_to_edge = clamp_to_edge

    def initialize_rgba(self, size):

        format = GL.GL_RGBA
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
                                depth_compare_mode=depth_compare_mode,
                                border_color = (1,1,1,1))

    def initialize_texture(self, size, format, iformat, tdtype, ncomp,
                           data=None, depth_compare_mode=False, border_color = (0, 0, 0, 0)):

        self.id = t = GL.glGenTextures(1)
        self.size = size
        self._check_maximum_texture_size(size)
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
            if self.is_cubemap:
                for face in range(6):
                    GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X+face,
                                    0, iformat, size[0], size[1], 0, format,
                                    tdtype, data)
            else:
                GL.glTexImage2D(gl_target, 0, iformat, size[0], size[1], 0, format,
                                tdtype, data)
        elif dim == 3:
            GL.glTexImage3D(gl_target, 0, iformat, size[0], size[1], size[2],
                            0, format, tdtype, data)

        GL.glTexParameterfv(gl_target, GL.GL_TEXTURE_BORDER_COLOR, border_color)
        clamp = GL.GL_CLAMP_TO_EDGE if self.is_cubemap or self.clamp_to_edge else GL.GL_CLAMP_TO_BORDER
        GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_S, clamp)
        if dim >= 2:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_T, clamp)
        if dim >= 3:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_WRAP_R, clamp)

        if self.linear_interpolation:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        else:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

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

    @property
    def gl_target(self):
        if self.is_cubemap:
            return GL.GL_TEXTURE_CUBE_MAP
        else:
            return (GL.GL_TEXTURE_1D, GL.GL_TEXTURE_2D, GL.GL_TEXTURE_3D)[self.dimension - 1]

    def set_linear_interpolation(self, linear):
        '''Has side effect of binding texture.'''
        if linear == self.linear_interpolation:
            return
        self.linear_interpolation = linear
        gl_target = self.gl_target
        GL.glBindTexture(gl_target, self.id)
        if linear:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        else:
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(gl_target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

    def _check_maximum_texture_size(self, size):
        if not hasattr(Texture, 'MAX_TEXTURE_SIZE'):
            Texture.MAX_TEXTURE_SIZE = GL.glGetInteger(GL.GL_MAX_TEXTURE_SIZE)
        max_size = Texture.MAX_TEXTURE_SIZE
        for s in size:
            if s > max_size:
                raise OpenGLError('Texture size (%s) exceeds OpenGL driver maximum %d' %
                                  (','.join(str(s) for s in size), max_size))

    def __del__(self):
        if self.id is not None:
            raise OpenGLError('OpenGL texture was not deleted before graphics.Texture destroyed')

    def delete_texture(self):
        'Delete the OpenGL texture.'
        if self.id is not None:
            GL.glDeleteTextures((self.id,))
            self.id = None

    def bind_texture(self, tex_unit=None):
        'Bind the OpenGL texture.'
        data = self.data
        if self.data is not None:
            self.fill_opengl_texture()
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

    def reload_texture(self, data, now = False):
        '''
        Replace the texture values in texture with OpenGL id using numpy
        array data.  The data is interpreted the same as for the Texture
        constructor data argument.
        '''
        # PyOpenGL 3.1.5 leaks memory if data not contiguous, PyOpenGL github issue #47.
        d = data if data.flags['C_CONTIGUOUS'] else data.copy()
        self.data = d
        if now:
            self.fill_opengl_texture()

    def fill_opengl_texture(self):
        data = self.data
        self.data = None
        if self.id is not None and tuple(data.shape) != self._array_shape or data.dtype != self._numpy_dtype:
            self.delete_texture()
        if self.id is None:
            size = tuple(data.shape[self.dimension - 1::-1])
            format, iformat, tdtype, ncomp = self.texture_format(data)
            self.initialize_texture(size, format, iformat, tdtype, ncomp, data)
        else:
            self._fill_texture(data)
        self._numpy_dtype = data.dtype
        self._array_shape = tuple(data.shape)

    def _fill_texture(self, data):
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
        level = 0
        xoffset = yoffset = zoffset = 0
        if dim == 1:
            GL.glTexSubImage1D(gl_target, level, xoffset, size[0],
                               format, tdtype, data)
        elif dim == 2:
            GL.glTexSubImage2D(gl_target, level, xoffset, yoffset, size[0], size[1],
                               format, tdtype, data)
        elif dim == 3:
            GL.glTexSubImage3D(gl_target, level, xoffset, yoffset, zoffset,
                               size[0], size[1], size[2],
                               format, tdtype, data)
        GL.glBindTexture(gl_target, 0)

    def texture_format(self, data):
        '''
        Return the OpenGL texture format, internal format, and texture
        value type that will be used by the glTexImageNd() function when
        creating a texture from a numpy array of colors.
        '''
        dim = self.dimension
        dtype = data.dtype
        from numpy import int8, uint8, int16, uint16, float32
        if dim == 2 and len(data.shape) == dim and dtype == uint32:
            format = GL.GL_RGBA
            iformat = GL.GL_RGBA8
            tdtype = GL.GL_UNSIGNED_BYTE
            ncomp = 4
            return format, iformat, tdtype, ncomp

        ncomp = data.shape[dim] if len(data.shape) > dim else 1

        format = {1: GL.GL_RED, 2: GL.GL_RG,
                  3: GL.GL_RGB, 4: GL.GL_RGBA}[ncomp]
        if dtype == int8:
            tdtype = GL.GL_BYTE
            iformat = {1: GL.GL_R8_SNORM, 2: GL.GL_RG8_SNORM,
                       3: GL.GL_RGB8_SNORM, 4: GL.GL_RGBA8_SNORM}[ncomp]
        elif dtype == uint8:
            tdtype = GL.GL_UNSIGNED_BYTE
            iformat = {1: GL.GL_R8, 2: GL.GL_RG8,
                       3: GL.GL_RGB8, 4: GL.GL_RGBA8}[ncomp]
        elif dtype == int16:
            tdtype = GL.GL_SHORT
            iformat = {1: GL.GL_R16_SNORM, 2: GL.GL_RG16_SNORM,
                       3: GL.GL_RGB16_SNORM, 4: GL.GL_RGBA16_SNORM}[ncomp]
        elif dtype == uint16:
            tdtype = GL.GL_UNSIGNED_SHORT
            iformat = {1: GL.GL_R16, 2: GL.GL_RG16,
                       3: GL.GL_RGB16, 4: GL.GL_RGBA16}[ncomp]
        elif dtype == float32:
            tdtype = GL.GL_FLOAT
            iformat = {1: GL.GL_R32F, 2: GL.GL_RG32F,
                       3: GL.GL_RGB32F, 4: GL.GL_RGBA32F}[ncomp]
        else:
            raise TypeError('Texture value type %s not supported' % str(dtype))
        return format, iformat, tdtype, ncomp

    def read_texture_data(self):
        '''
        The data is read back to a numpy array as uint8 values using the
        same array shape used to fill the texture.
        '''
        from numpy import zeros, uint8
        data = zeros(self._array_shape, uint8)
        format, iformat, tdtype, ncomp = self.texture_format(data)
        gl_target = self.gl_target
        GL.glBindTexture(gl_target, self.id)
        level = 0
        GL.glGetTexImage(gl_target, level, format, tdtype, data)
        GL.glBindTexture(gl_target, 0)
        return data

    def normalization(self):
        '''
        Scale factor for converting texture values to normalized values,
        0-1 for unsigned integer, -1 to 1 for signed integer.
        '''
        if not hasattr(self, '_normalizations'):
            from numpy import int8, uint8, int16, uint16, float32
            self._normalizations = {
                int8: 1/127,
                uint8: 1/255,
                int16: 1/32767,
                uint16: 1/65535,
                float32: 1
                }
        dtype = self._numpy_dtype if self.data is None else self.data.dtype
        return self._normalizations[dtype.type]


class TextureWindow:
    '''
    Draw a texture on a full window rectangle. Don't test or write depth buffer.
    '''
    def __init__(self, render):

        # Must have vao bound before compiling shader.
        self.vao = vao = Bindings('texture window', render.opengl_context)
        vao.activate()

        self.vertex_buf = vb = Buffer(VERTEX_BUFFER)
        self.tex_coord_buf = tcb = Buffer(TEXTURE_COORDS_BUFFER)
        self.element_buf = eb = Buffer(ELEMENT_BUFFER)

        from numpy import array, float32, int32
        va = array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float32)
        tc = array([(0, 0), (1, 0), (1, 1), (0, 1)], float32)
        ta = array([(0, 1, 2), (0, 2, 3)], int32)

        self.vertex_buf.update_buffer_data(va)
        self.tex_coord_buf.update_buffer_data(tc)
        self.element_buf.update_buffer_data(ta)

        self._bind_shader_variables() # bind changed buffers

    def __del__(self):
        if self.vao is not None:
            raise OpenGLError('graphics.TextureWindow delete() not called')

    def delete(self):
        if self.vao is None:
            return
        self.vao.delete_bindings()
        self.vao = None
        for b in (self.vertex_buf, self.tex_coord_buf, self.element_buf):
            b.delete_buffer()
        self.vertex_buf = self.tex_coord_buf = self.element_buf = None

    def _bind_shader_variables(self):
        vao = self.vao
        for b in (self.vertex_buf, self.tex_coord_buf, self.element_buf):
            vao.bind_shader_variable(b)

    def activate(self):
        self.vao.activate()

    def draw(self, blend = False):
        if blend:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glDepthMask(False)   # Don't overwrite depth buffer
        GL.glDisable(GL.GL_DEPTH_TEST) # Don't test depth buffer.

        offset, count = 0, 6
        eb = self.element_buf
        eb.draw_elements(eb.triangles, offset = offset, count = count)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(True)

        if blend:
            GL.glDisable(GL.GL_BLEND)

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

def pyopengl_string_list(strings):
    import ctypes
    bufs = [ctypes.create_string_buffer(name.encode()) for name in strings]
    from numpy import array
    bpointers = array([ctypes.addressof(b) for b in bufs])
    bpa = bpointers.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    bpa._string_buffers = bufs     # Keep string buffers from being released
    bpa._pointer_array = bpointers # Keep numpy array from being released
    return bpa

def pyopengl_null():
    import ctypes
    return ctypes.c_void_p(0)

class OffScreenRenderingContext:

    def __init__(self, width = 512, height = 512):
        self.width = width
        self.height = height
        import ctypes
        _initialize_pyopengl(offscreen = True)
        import OpenGL
        from OpenGL import osmesa
        from OpenGL import GL, arrays, platform, error
        # To use OSMesa with PyOpenGL requires environment variable PYOPENGL_PLATFORM=osmesa
        # Also will need libOSMesa in dlopen library path.
        attribs = [
            osmesa.OSMESA_FORMAT, osmesa.OSMESA_RGBA,
            osmesa.OSMESA_DEPTH_BITS, 32,
            # osmesa.OSMESA_STENCIL_BITS, 8,
            osmesa.OSMESA_PROFILE, osmesa.OSMESA_CORE_PROFILE,
            osmesa.OSMESA_CONTEXT_MAJOR_VERSION, 3,
            osmesa.OSMESA_CONTEXT_MINOR_VERSION, 3,
            0  # must end with zero
        ]
        attribs = (ctypes.c_int * len(attribs))(*attribs)
        try:
            self.context = osmesa.OSMesaCreateContextAttribs(attribs, None)
        except error.NullFunctionError:
            raise OpenGLError('Need OSMesa version 12.0 or newer for OpenGL Core Context API.')
        if not self.context:
            raise OpenGLError('OSMesa needs to be configured with --enable-gallium-osmesa for OpenGL Core Context support.')
        buf = arrays.GLubyteArray.zeros((height, width, 4))
        self.buffer = buf
        # call make_current to induce exception if an older Mesa
        self.make_current()

        # Keep track of allocated framebuffers and vertex array objects.
        from weakref import WeakSet
        self._framebuffers = WeakSet() # Set of Framebuffer objects
        self._bindings = WeakSet()     # Set of Bindings objects

        # Draw target for default framebuffer
        self.default_draw_target = GL.GL_FRONT

        # compatibility with OpenGLContext
        self._framebuffer_color_bits = 8

    def make_current(self):
        from OpenGL import GL, arrays, platform
        from OpenGL import osmesa
        assert(osmesa.OSMesaMakeCurrent(self.context, self.buffer, GL.GL_UNSIGNED_BYTE, self.width, self.height))
        assert(platform.CurrentContextIsValid())
        return True

    def swap_buffers(self):
        pass

    def pixel_scale(self):
        # Ratio Qt pixel size to OpenGL pixel size.
        return 1
