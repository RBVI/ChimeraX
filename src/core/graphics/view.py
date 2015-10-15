# vi: set expandtab shiftwidth=4 softtabstop=4:
'''
View
====
'''


class View:
    '''
    A View is the graphics windows that shows 3-dimensional drawings.
    It manages the camera and renders the drawing when needed.
    '''
    def __init__(self, drawing, *, window_size = (256,256), opengl_context = None,
                 trigger_set = None):

        self.triggers = trigger_set
        self.drawing = drawing
        self.window_size = window_size		# pixels
        self._opengl_context = None
        self._render = None

        # Red, green, blue, opacity, 0-1 range.
        self._background_rgba = (0, 0, 0, 1)

        # Create camera
        from .camera import Camera
        self.camera = Camera()
        '''The camera controlling the vantage shown in the graphics window.'''

        if opengl_context:
            self.initialize_context(opengl_context)
        self._opengl_initialized = False
        self._near_far_pad = 0.01		# Extra near-far clip plane spacing.
        self._shadows = False
        self._shadow_map_size = 2048
        self._shadow_depth_bias = 0.005
        self._multishadow = 0                   # Number of shadows
        self._multishadow_map_size = 128
        self._multishadow_depth_bias = 0.05
        self._multishadow_dir = None
        self._multishadow_transforms = []
        self._multishadow_depth = None
        self._multishadow_update_needed = False

        self.silhouettes = False
        self.silhouette_thickness = 1           # pixels
        self.silhouette_color = (0, 0, 0, 1)    # black
        self.silhouette_depth_jump = 0.03       # fraction of scene depth

        self.frame_number = 1
        self.redraw_needed = True
        self._time_graphics = False
        self.update_lighting = False

        self._overlays = []
        self._2d_overlays = []

        from numpy import array, float32
        self._center_of_rotation = array((0, 0, 0), float32)
        self._update_center_of_rotation = False

        self._drawing_manager = dm = _RedrawNeeded()
        if trigger_set:
            self.drawing.set_redraw_callback(dm)

    def initialize_context(self, oc):
        if self._opengl_context is not None:
            raise ValueError("OpenGL context is alread set")
        self._opengl_context = oc
        from .opengl import Render
        self._render = Render()
        self.depth_cue = True

    def opengl_context(self):
        return self._opengl_context

    def opengl_version(self):
        '''Return the OpenGL version as a string.'''
        self._use_opengl()
        return self._render.opengl_version()

    def _use_opengl(self):
        if self._opengl_context is None:
            raise RuntimeError("running without graphics")
        self._opengl_context.make_current()
        self._initialize_opengl()

    def _initialize_opengl(self):

        if self._opengl_initialized:
            return
        self._opengl_initialized = True

        r = self._render
        r.set_background_color(self.background_color)
        r.enable_depth_test(True)

        w, h = self.window_size
        r.initialize_opengl(w, h)

    def draw(self, camera = None, drawings = None,
             check_for_changes = True, swap_buffers = True):
        '''
        Draw the scene.
        '''
        self._use_opengl()

        if check_for_changes:
            self.check_for_drawing_change()

        if camera is None:
            camera = self.camera

        mdraw = [self.drawing] if drawings is None else drawings

        r = self._render
        if self.shadows:
            # Light direction in camera coords
            kl = r.lighting.key_light_direction
            # Light direction in scene coords.
            lightdir = camera.position.apply_without_translation(kl)
            stf = self._use_shadow_map(lightdir, drawings)
        if self.multishadow > 0:
            mstf, msdepth \
                = self._use_multishadow_map(self._multishadow_directions(),
                                            drawings)

        r.set_background_color(self.background_color)

        if self.update_lighting:
            self.update_lighting = False
            r.set_shader_lighting_parameters()

        if drawings is None:
            any_selected = self.any_drawing_selected()
        else:
            any_selected = True

        r.set_frame_number(self.frame_number)
        perspective_near_far_ratio = 2
        from .drawing import (draw_depth, draw_drawings, draw_outline,
                              draw_overlays, draw_2d_overlays)
        for vnum in range(camera.number_of_views()):
            camera.set_render_target(vnum, r)
            if self.silhouettes:
                r.start_silhouette_drawing()
            r.draw_background()
            if mdraw:
                perspective_near_far_ratio \
                    = self._update_projection(vnum, camera=camera)
                cp = camera.get_position(vnum)
                cpinv = cp.inverse()
                if self.shadows and stf is not None:
                    r.set_shadow_transform(stf * cp)
                if self.multishadow > 0 and mstf is not None:
                    r.set_multishadow_transforms(mstf, cp, msdepth)
                    # Initial depth pass optimization to avoid lighting
                    # calculation on hidden geometry
                    draw_depth(r, cpinv, mdraw)
                    r.allow_equal_depth(True)
                self._start_timing()
                draw_drawings(r, cpinv, mdraw)
                self._finish_timing()
                if self.multishadow > 0:
                    r.allow_equal_depth(False)
                if any_selected:
                    draw_outline(r, cpinv, mdraw)
            if self.silhouettes:
                r.finish_silhouette_drawing(self.silhouette_thickness,
                                            self.silhouette_color,
                                            self.silhouette_depth_jump,
                                            perspective_near_far_ratio)

        camera.combine_rendered_camera_views(r)

        if self._overlays:
            draw_overlays(self._overlays, r)

        if self._2d_overlays:
            draw_2d_overlays(self._2d_overlays, r)

        if swap_buffers:
            if self.camera.mode.do_swap_buffers():
                self._opengl_context.swap_buffers()
            self.redraw_needed = False

    def check_for_drawing_change(self):
        trig = self.triggers
        if trig:
            trig.activate_trigger('graphics update', self)

        c = self.camera
        dm = self._drawing_manager
        draw = self.redraw_needed or c.redraw_needed or dm.redraw_needed
        if not draw:
            return False

        if dm.shape_changed:
            if trig:
                trig.activate_trigger('shape changed', self)	# Used for updating pseudobond graphics
            self._update_center_of_rotation = True

        if dm.redraw_needed and dm.shape_changed and self.multishadow > 0:
            self._multishadow_update_needed = True

        c.redraw_needed = False
        dm.redraw_needed = False
        dm.shape_changed = False

        self.redraw_needed = True

        return True

    def get_background_color(self):
        return self._background_rgba

    def set_background_color(self, rgba):
        self._background_rgba = tuple(rgba)
        self.redraw_needed = True
    background_color = property(get_background_color, set_background_color)
    '''Background color as R, G, B, A values in 0-1 range.'''

    def lighting(self):
        '''Lighting parameters.'''
        return self._render.lighting

    def material(self):
        '''Material reflectivity parameters.'''
        return self._render.material

    def get_depth_cue(self):
        r = self._render
        return bool(r.enable_capabilities & r.SHADER_DEPTH_CUE)
    def set_depth_cue(self, enable):
        r = self._render
        if enable:
            r.enable_capabilities |= r.SHADER_DEPTH_CUE
        else:
            r.enable_capabilities &= ~r.SHADER_DEPTH_CUE
        self.redraw_needed = True
    depth_cue = property(get_depth_cue, set_depth_cue)
    '''Is dimming with depth enabled. Boolean value.'''

    def get_shadows(self):
        return self._shadows

    def set_shadows(self, onoff):
        onoff = bool(onoff)
        if onoff == self._shadows:
            return
        self._shadows = onoff
        r = self._render
        if onoff:
            r.enable_capabilities |= r.SHADER_SHADOWS
        else:
            r.enable_capabilities &= ~r.SHADER_SHADOWS
        self.redraw_needed = True
    shadows = property(get_shadows, set_shadows)
    '''Is a shadow cast by the key light enabled? Boolean value.'''

    def max_multishadow(self):
        return self._render.max_multishadows()
    def get_multishadow(self):
        return self._multishadow
    def set_multishadow(self, n):
        '''
        Specify the number of shadows to use for ambient shadowing,
        for example, 64 or 128.  To turn off ambient shadows specify 0
        shadows.  Shadows are cast from uniformly distributed directions.
        This is GPU intensive, each shadow requiring a texture lookup.
        '''
        self._multishadow = n
        r = self._render
        if n > 0:
            r.enable_capabilities |= r.SHADER_MULTISHADOW
        else:
            # TODO: free multishadow framebuffer.
            self._multishadow_transforms = []
            r.enable_capabilities &= ~r.SHADER_MULTISHADOW
        self.redraw_needed = True
    multishadow = property(get_multishadow, set_multishadow)

    def get_shadow_depth_bias(self):
        return self._shadow_depth_bias
    def set_shadow_depth_bias(self, bias):
        self._shadow_depth_bias = bias
        self.redraw_needed = True
    shadow_depth_bias = property(get_shadow_depth_bias, set_shadow_depth_bias)

    def get_multishadow_depth_bias(self):
        return self._multishadow_depth_bias
    def set_multishadow_depth_bias(self, bias):
        self._multishadow_depth_bias = bias
        self._multishadow_transforms = []
        self.redraw_needed = True
    multishadow_depth_bias = property(get_multishadow_depth_bias, set_multishadow_depth_bias)
    
    def get_shadow_map_size(self):
        return self._shadow_map_size
    def set_shadow_map_size(self, size):
        '''
        Set the size of the 2-d texture for casting shadows.
        Typical values are 1024, 2048, 4096.
        Larger sizes give shadows with smoother edges.
        '''
        if size != self._shadow_map_size:
            self._shadow_map_size = size
            self.redraw_needed = True
    shadow_map_size = property(get_shadow_map_size, set_shadow_map_size)
    
    def get_multishadow_map_size(self):
        return self._multishadow_map_size
    def set_multishadow_map_size(self, size):
        '''
        Set the size of the 2-d texture for casting shadows.
        Small values (128, 256) give nicer smoother appearance.
        '''
        if size != self._multishadow_map_size:
            self._multishadow_map_size = size
            self._multishadow_transforms = []   # Cause shadow recomputation
            self.redraw_needed = True
    multishadow_map_size = property(get_multishadow_map_size, set_multishadow_map_size)

    def add_overlay(self, overlay):
        '''
        Overlays are Drawings rendered after the normal scene is shown.
        They are used for effects such as motion blur or cross fade that
        blend the current rendered scene with a previous rendered scene.
        '''
        overlay.set_redraw_callback(self._drawing_manager)
        self._overlays.append(overlay)
        self.redraw_needed = True

    def overlays(self):
        '''The current list of overlay Drawings.'''
        return self._overlays

    def remove_overlays(self, overlays=None):
        '''Remove the specified overlay Drawings.'''
        if overlays is None:
            overlays = self._overlays
        for o in overlays:
            o.delete()
        oset = set(overlays)
        self._overlays = [o for o in self._overlays if o not in oset]
        self.redraw_needed = True

    def add_2d_overlay(self, overlay):
        '''
        Overlays are Drawings rendered after the normal scene is shown.
        They are used for effects such as motion blur or cross fade that
        blend the current rendered scene with a previous rendered scene.
        '''
        overlay.set_redraw_callback(self._drawing_manager)
        self._2d_overlays.append(overlay)
        self.redraw_needed = True

    def twod_overlays(self):
        '''The current list of overlay Drawings.'''
        return self._2d_overlays

    def remove_2d_overlays(self, overlays=None):
        '''Remove the specified overlay Drawings.'''
        if overlays is None:
            overlays = self._2d_overlays
        for o in overlays:
            o.delete()
        oset = set(overlays)
        self._2d_overlays = [o for o in self._overlays if o not in oset]
        self.redraw_needed = True

    def image(self, width=None, height=None, supersample=None, camera=None,
              drawings=None):
        '''Capture an image of the current scene. A PIL image is returned.'''

        self._use_opengl()

        w, h = self._window_size_matching_aspect(width, height)

        from .opengl import Framebuffer
        fb = Framebuffer(w, h)
        if not fb.valid():
            return None         # Image size exceeds framebuffer limits

        r = self._render
        r.push_framebuffer(fb)

        if camera is None:
            if drawings:
                from .camera import camera_framing_drawings
                c = camera_framing_drawings(drawings)
                if c is None:
                    return None         # Drawings not showing anything
            else:
                c = self.camera
        else:
            c = camera

        if supersample is None:
            self.draw(c, drawings, swap_buffers = False)
            rgba = r.frame_buffer_image(w, h)
        else:
            from numpy import zeros, float32, uint8
            srgba = zeros((h, w, 4), float32)
            n = supersample
            s = 1.0 / n
            s0 = -0.5 + 0.5 * s
            for i in range(n):
                for j in range(n):
                    c.pixel_shift = (s0 + i * s, s0 + j * s)
                    self.draw(c, drawings, swap_buffers = False)
                    srgba += r.frame_buffer_image(w, h)
            c.pixel_shift = (0, 0)
            srgba /= n * n
            # third index 0, 1, 2, 3 is r, g, b, a
            rgba = srgba.astype(uint8)
        r.pop_framebuffer()
        fb.delete()

        # Flip y-axis since PIL image has row 0 at top,
        # opengl has row 0 at bottom.
        from PIL import Image
        pi = Image.fromarray(rgba[::-1, :, :3])
        return pi

    def frame_buffer_rgba(self):
        '''
        Return a numpy array of R, G, B, A values of the currently
        rendered scene.  This is used for blending effects such as motion
        blur and cross fades.
        '''
        w, h = self.window_size
        rgba = self._render.frame_buffer_image(w, h, front_buffer = True)
        return rgba

    def resize(self, width, height):
        '''
        This is called when the graphics window was resized by the
        user and causes the OpenGL rendering to use the specified new
        window size.
        '''
        new_size = (width, height)
        if self.window_size == new_size and self._opengl_initialized:
            return
        self.window_size = new_size
        if self._opengl_initialized:
            fb = self._render.default_framebuffer()
            fb.width, fb.height = width, height
            fb.viewport = (0, 0, width, height)

    def _window_size_matching_aspect(self, width, height):
        w, h = width, height
        vw, vh = self.window_size
        if w is not None and h is not None:
            return (w, h)
        elif w is not None:
            # Choose height to match window aspect ratio.
            return (w, (vh * w) // vw)
        elif height is not None:
            # Choose width to match window aspect ratio.
            return ((vw * h) // vh, h)
        return (vw, vh)

    def report_framerate(self, report_rate, monitor_period=1.0, _minimum_render_time=None):
        '''
        Report a status message giving the current rendering rate in
        frames per second.  This is computed without the vertical sync
        which normally limits the frame rate to typically 60 frames
        per second.  The minimum drawing time used over a one second
        interval is used. The report_rate function is called with
        the frame rate in frames per second.
        '''
        if _minimum_render_time is None:
            self._framerate_callback = report_rate
            from time import time
            self._time_graphics = time() + monitor_period
            self.minimum_render_time = None
            self.render_start_time = None
            self.redraw_needed = True
        else:
            self._time_graphics = 0
            self._framerate_callback(1.0/_minimum_render_time)

    def _start_timing(self):
        if self._time_graphics:
            self.finish_rendering()
            from time import time
            self.render_start_time = time()

    def _finish_timing(self):
        if self._time_graphics:
            self.finish_rendering()
            from time import time
            t = time()
            rt = t - self.render_start_time
            mint = self.minimum_render_time
            if mint is None or rt < mint:
                self.minimum_render_time = mint = rt
            if t > self._time_graphics:
                self.report_framerate(None, _minimum_render_time = mint)
            else:
                self.redraw_needed = True

    def finish_rendering(self):
        '''
        Force the graphics pipeline to complete all requested drawing.
        This can slow down rendering but is used by display devices
        such as Oculus Rift goggles to reduce latency time between head
        tracking and graphics update.
        '''
        self._render.finish_rendering()

    def _multishadow_directions(self):

        directions = self._multishadow_dir
        if directions is None or len(directions) != self.multishadow:
            n = self.multishadow    # requested number of directions
            from ..geometry import sphere
            self._multishadow_dir = directions = sphere.sphere_points(n)
        return directions

    def _use_shadow_map(self, light_direction, drawings):

        r = self._render

        # Compute drawing bounds so shadow map can cover all drawings.
        center, radius, bdrawings = _drawing_bounds(drawings, self.drawing)
        if center is None:
            return None

        # Compute shadow map depth texture
        size = self.shadow_map_size
        r.start_rendering_shadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # Compute light view and scene to shadow map transforms
        bias = self._shadow_depth_bias
        lvinv, stf = r.shadow_transforms(light_direction, center, radius, bias)
        from .drawing import draw_drawings
        draw_drawings(r, lvinv, bdrawings)

        shadow_map = r.finish_rendering_shadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.shadow_texture_unit)

        return stf      # Scene to shadow map texture coordinates

    def _use_multishadow_map(self, light_directions, drawings):

        if self._multishadow_update_needed:
            self._multishadow_transforms = []
            self._multishadow_update_needed = False

        r = self._render
        if len(self._multishadow_transforms) == len(light_directions):
            # Bind shadow map for subsequent rendering of shadows.
            dt = r.multishadow_map_framebuffer.depth_texture
            dt.bind_texture(r.multishadow_texture_unit)
            return self._multishadow_transforms, self._multishadow_depth

        # Compute drawing bounds so shadow map can cover all drawings.
        center, radius, bdrawings = _drawing_bounds(drawings, self.drawing)
        if center is None:
            return None, None

        # Compute shadow map depth texture
        size = self.multishadow_map_size
        r.start_rendering_multishadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # TODO: Don't run multishadow fragment shader when computing shadow map
        #    -- very expensive.
        mstf = []
        nl = len(light_directions)
        from .drawing import draw_drawings
        from math import ceil, sqrt
        d = int(ceil(sqrt(nl)))     # Number of subtextures along each axis
        s = size // d               # Subtexture size.
        bias = self._multishadow_depth_bias
        for l in range(nl):
            x, y = (l % d), (l // d)
            r.set_viewport(x * s, y * s, s, s)
            lvinv, tf = r.shadow_transforms(light_directions[l], center, radius, bias)
            mstf.append(tf)
            draw_drawings(r, lvinv, bdrawings)

        shadow_map = r.finish_rendering_multishadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.multishadow_texture_unit)

        # TODO: Clear shadow cache whenever scene changes
        self._multishadow_transforms = mstf
        self._multishadow_depth = msd = 2 * radius
#        r.set_multishadow_transforms(mstf, None, msd)
        return mstf, msd      # Scene to shadow map texture coordinates

    def drawing_bounds(self):
        '''Return bounds of drawing, displayed part only.'''
        dm = self._drawing_manager
        b = dm.cached_drawing_bounds
        if b is None or self.check_for_drawing_change():
            dm.cached_drawing_bounds = b = self.drawing.bounds()
        return b

    def any_drawing_selected(self):
        '''Is anything selected.'''
        dm = self._drawing_manager
        s = dm.cached_any_part_selected
        if s is None:
            dm.cached_any_part_selected = s = self.drawing.any_part_selected()
        return s

    def initial_camera_view(self):
        '''Set the camera position to show all displayed drawings,
        looking down the z axis.'''
        b = self.drawing_bounds()
        if b is None:
            return
        self.camera.initialize_view(b.center(), b.width())
        self.center_of_rotation = b.center()

    def view_all(self):
        '''Adjust the camera to show all displayed drawings using the
        current view direction.'''
        b = self.drawing_bounds()
        if b is None:
            return
        shift = self.camera.view_all(b.center(), b.width())
        self.translate(-shift)

    def _get_cofr(self):
        if self._update_center_of_rotation:
            self._update_center_of_rotation = False
            cofr = self._compute_center_of_rotation()
            if not cofr is None:
                self._center_of_rotation = cofr
        return self._center_of_rotation
    def _set_cofr(self, cofr):
        self._center_of_rotation = cofr
        self._update_center_of_rotation = False
    center_of_rotation = property(_get_cofr, _set_cofr)

    def _compute_center_of_rotation(self):
        '''
        Compute the center of rotation of displayed drawings.
        Use bounding box center if zoomed out, or the front center
        point if zoomed in.
        '''
        self._update_center = False
        b = self.drawing_bounds()
        if b is None:
            return
        vw = self.camera.view_width(b.center())
        if vw >= b.width():
            # Use center of drawings for zoomed out views
            cr = b.center()
        else:
            # Use front center point for zoomed in views
            cr = self._front_center_point()
            if cr is None:
                return None
        return cr

    def _front_center_point(self):
        w, h = self.window_size
        p = self.first_intercept(0.5 * w, 0.5 * h)
        return p.position if p else None

    def first_intercept(self, win_x, win_y):
        '''
        Return a Pick object for the front-most object below the given
        screen window position (specified in pixels).  This Pick object will
        have an attribute position giving the point where the intercept occurs.
        This is used when hovering the mouse over an object (e.g. an atom)
        to get a description of that object.
        '''
        xyz1, xyz2 = self.clip_plane_points(win_x, win_y)
        p = self.drawing.first_intercept(xyz1, xyz2, exclude='is_outline_box')
        if p is None:
            return None
        f = p.distance
        p.position = (1.0 - f) * xyz1 + f * xyz2
        return p

    def _update_projection(self, view_num=None, camera=None):

        r = self._render
        ww, wh = r.render_size()
        if ww == 0 or wh == 0:
            return

        c = self.camera if camera is None else camera
        near, far = self._near_far_clip(c, view_num)
        pm = c.projection_matrix((near, far), view_num, (ww, wh))
        r.set_projection_matrix(pm)
        r.set_near_far_clip(near, far)

        return near / far

    def _near_far_clip(self, camera, view_num):
        # Return the near and far clip plane distances.

        cp = camera.get_position(view_num).origin()
        vd = camera.view_direction(view_num)
        b = self.drawing_bounds()
        if b is None:
            return 0.001, 1  # Nothing shown
        d = sum((b.center() - cp) * vd)         # camera to center of drawings
        r = (0.5 + self._near_far_pad) * b.width()
        near, far = (d - r, d + r)

        # Clamp near clip > 0.
        near_min = 0.001 * (far - near) if far > near else 1
        near = max(near, near_min)
        if far <= near:
            far = 2 * near

        return (near, far)

    def clip_plane_points(self, window_x, window_y, camera=None,
                          view_num=None):
        '''
        Return two scene points at the near and far clip planes at
        the specified window pixel position.  The points are in scene
        coordinates.  '''
        c = camera if camera else self.camera
        nf = self._near_far_clip(c, view_num)
        scene_pts = c.clip_plane_points(window_x, window_y, self.window_size,
                                        nf, self._render)
        return scene_pts

    def rotate(self, axis, angle, drawings=None):
        '''
        Move camera to simulate a rotation of drawings about current
        rotation center.  Axis is in scene coordinates and angle is
        in degrees.
        '''
        if drawings:
            from ..geometry import bounds
            b = bounds.union_bounds(d.bounds() for d in drawings)
            if b is None:
                return
            center = b.center()
        else:
            center = self.center_of_rotation
        from ..geometry import place
        r = place.rotation(axis, angle, center)
        self.move(r, drawings)

    def translate(self, shift, drawings=None):
        '''Move camera to simulate a translation of drawings.  Translation
        is in scene coordinates.'''
        self._update_center_of_rotation = True
        from ..geometry import place
        t = place.translation(shift)
        self.move(t, drawings)

    def move(self, tf, drawings=None):
        '''Move camera to simulate a motion of drawings.'''
        if drawings is None:
            c = self.camera
            c.position = tf.inverse() * c.position
        else:
            for d in drawings:
                d.position = tf * d.position

        self.redraw_needed = True

    def pixel_size(self, p=None):
        '''Return the pixel size in scene length units at point p in
        the scene.'''
        if p is None:
            p = self.center_of_rotation
        return self.camera.pixel_size(p, self.window_size)


class OpenGLContext:
    '''
    OpenGL context used by View for drawing.  This should be subclassed
    to provide window system specific opengl context methods.
    '''
    def make_current(self):
        '''Make the OpenGL context active.'''
        pass

    def swap_buffers(self):
        '''Swap back and front OpenGL buffers.'''
        pass


class _RedrawNeeded:

    def __init__(self):
        self.redraw_needed = False
        self.shape_changed = True
        self.cached_drawing_bounds = None
        self.cached_any_part_selected = None

    def __call__(self, shape_changed=False, selection_changed=False):
        self.redraw_needed = True
        if shape_changed:
            self.shape_changed = True
            self.cached_drawing_bounds = None
        if selection_changed:
            self.cached_any_part_selected = None


def _drawing_bounds(drawings, open_drawing):
    if drawings is None:
        b = open_drawing.bounds()
        bdrawings = [open_drawing]
    else:
        from ..geometry import bounds
        b = bounds.union_bounds(d.bounds() for d in drawings)
        bdrawings = drawings
    center = None if b is None else b.center()
    radius = None if b is None else b.radius()
    return center, radius, bdrawings
