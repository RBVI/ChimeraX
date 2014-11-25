'''
View
====
'''

class View:
    '''
    A View is the graphics windows that shows 3-dimensional drawings.
    It manages the camera and renders the drawing when needed.
    '''
    def __init__(self, drawing, window_size, opengl_context, log):

        self.drawing = drawing
        self.log = log
        self.window_size = window_size		# pixels
        self._opengl_context = opengl_context

        self._background_rgba = (0,0,0,1)        # Red, green, blue, opacity, 0-1 range.

        # Create camera
        from .camera import Camera
        self.camera = Camera()
        '''The camera controlling the vantage shown in the graphics window.'''

        from .opengl import Render
        self._render = Render()
        self._opengl_initialized = False
        self._shadows = False
        self.shadow_map_size = 2048
        self.multishadow = 0                    # Number of shadows
        self._multishadow_dir = None
        self._multishadow_transforms = []
        self._multishadow_depth = None
        self.silhouettes = False
        self.silhouette_thickness = 1           # pixels
        self.silhouette_color = (0,0,0,1)       # black
        self.silhouette_depth_jump = 0.01       # fraction of scene depth

        self.frame_number = 1
        self.redraw_needed = False
        self._time_graphics = False
        self.update_lighting = False
        self._block_redraw_count = 0
        self._new_frame_callbacks = []
        self._rendered_callbacks = []
        self._shape_changed_callbacks = []

        self._overlays = []

        from numpy import array, float32
        self.center_of_rotation = array((0,0,0), float32)
        self._update_center = True

        self._drawing_manager = dm = _Redraw_Needed()
        drawing.set_redraw_callback(dm)

    def _initialize_opengl(self):

        if self._opengl_initialized:
            return
        self._opengl_initialized = True

        r = self._render
        r.set_background_color(self.background_color)
        r.enable_depth_test(True)

        w,h = self.window_size
        r.initialize_opengl(w,h)

    def opengl_version(self):
        '''Return the OpenGL version as a string.'''
        return self._render.opengl_version()

    def _use_opengl(self):
        self._opengl_context.make_current()
        self._initialize_opengl()

    def draw(self):
        '''Draw the scene.'''
        self._use_opengl()
        self._draw_scene()
        if self.camera.mode.do_swap_buffers():
            self._opengl_context.swap_buffers()
        self.frame_number += 1

    def get_background_color(self):
        return self._background_rgba
    def set_background_color(self, rgba):
        self._background_rgba = tuple(rgba)
        self.redraw_needed = True
    background_color = property(get_background_color, set_background_color)
    '''Background color as R,G,B,A values in 0-1 range.'''

    def lighting(self):
        '''Lighting parameters.'''
        return self._render.lighting
    def material(self):
        '''Material reflectivity parameters.'''
        return self._render.material

    def enable_depth_cue(self, enable):
        '''Turn on or off dimming with depth.'''
        r = self._render
        if enable:
            r.enable_capabilities |= r.SHADER_DEPTH_CUE
        else:
            r.enable_capabilities &= ~r.SHADER_DEPTH_CUE
        self.redraw_needed = True
    def depth_cue_enabled(self):
        '''Is depth cue enabled. Boolean value.'''
        r = self._render
        return bool(r.enable_capabilities | r.SHADER_DEPTH_CUE)

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

    def set_multishadow(self, n):
        '''
        Specify the number of shadows to use for ambient shadowing, for example, 64 or 128.
        To turn off ambient shadows specify 0 shadows.  Shadows are cast from uniformly
        distributed directions.  This is GPU intensive, each shadow requiring a texture lookup.
        '''
        self.multishadow = n
        r = self._render
        if n > 0:
            r.enable_capabilities |= r.SHADER_MULTISHADOW
        else:
            # TODO: free multishadow framebuffer.
            self._multishadow_transforms = []
            r.enable_capabilities &= ~r.SHADER_MULTISHADOW
        self.redraw_needed = True

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

    def remove_overlays(self, overlays = None):
        '''Remove the specified overlay Drawings.'''
        if overlays is None:
            overlays = self._overlays
        for o in overlays:
            o.delete()
        oset = set(overlays)
        self._overlays = [o for o in self._overlays if not o in oset]
        self.redraw_needed = True

    def image(self, width = None, height = None, supersample = None, camera = None, drawings = None):
        '''Capture an image of the current scene. A PIL image is returned.'''
        self._use_opengl()

        w,h = self._window_size_matching_aspect(width, height)

        from .opengl import Framebuffer
        fb = Framebuffer(w,h)
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
            self._draw_scene(c, drawings)
            rgba = r.frame_buffer_image(w, h)
        else:
            from numpy import zeros, float32, uint8
            srgba = zeros((h,w,4), float32)
            n = supersample
            s = 1.0/n
            s0 = -0.5 + 0.5*s
            for i in range(n):
                for j in range(n):
                    c.pixel_shift = (s0 + i*s, s0 + j*s)
                    self._draw_scene(c, drawings)
                    srgba += r.frame_buffer_image(w,h)
            c.pixel_shift = (0,0)
            srgba /= n*n
            rgba = srgba.astype(uint8)            # third index 0,1,2,3 is r,g,b,a
        r.pop_framebuffer()
        fb.delete()

        # Flip y-axis since PIL image has row 0 at top, opengl has row 0 at bottom.
        from PIL import Image
        pi = Image.fromarray(rgba[::-1,:,:3])
        return pi

    def frame_buffer_rgba(self):
        '''
        Return a numpy array of R,G,B,A values of the currently rendered scene.
        This is used for blending effects such as motion blur and cross fades.
        '''
        w,h = self.window_size
        rgba = self._render.frame_buffer_image(w, h)
        return rgba

    def resize(self, width, height):
        '''
        This is called when the graphics window was resized by the user and causes
        the OpenGL rendering to use the specified new window size.
        '''
        self.window_size = (width, height)
        if self._opengl_initialized:
            from .opengl import default_framebuffer
            fb = default_framebuffer()
            fb.width, fb.height = width,height
            fb.viewport = (0,0,width,height)

    def _window_size_matching_aspect(self, width, height):
        w,h = width, height
        vw,vh = self.window_size
        if not w is None and not h is None:
            return (w,h)
        elif not w is None:
            return (w, (vh*w)//vw)     # Choose height to match window aspect ratio.
        elif not height is None:
            return ((vw*h)//vh, h)     # Choose width to match window aspect ratio.
        return (vw,vh)

    def draw_if_changed(self):
        '''Redraw the scene if any changes have occured.'''

        if self._block_redraw_count > 0:
            return False
        
        self._block_redraw()	# Avoid redrawing during callbacks of the current redraw.
        try:
            return self._draw_if_changed()
        finally:
            self._unblock_redraw()

        print('draw if changed')
        return False

    def _draw_if_changed(self):
        for cb in self._new_frame_callbacks:
            try:
                cb()
            except:
                import traceback
                self.log.show_warning('new frame callback rasied error\n'
                                      + traceback.format_exc())
                self.remove_new_frame_callback(cb)
                

        c = self.camera
        dm = self._drawing_manager
        draw = self.redraw_needed or c.redraw_needed or dm.redraw_needed
        if not draw:
            return False

        if dm.shape_changed:
            for cb in tuple(self._shape_changed_callbacks):
                try:
                    cb()
                except:
                    import traceback
                    self.log.show_warning('shape changed callback rasied error\n'
                                      + traceback.format_exc())
                    self.remove_shape_changed_callback(cb)
                
        if dm.redraw_needed and dm.shape_changed and self.multishadow > 0:
            # Force recomputation of ambient shadows since shape changed.
            self._multishadow_transforms = []

        self.redraw_needed = False
        c.redraw_needed = False
        dm.redraw_needed = False
        dm.shape_changed = False
        self.draw()
        for cb in self._rendered_callbacks:
            try:
                cb()
            except:
                import traceback
                self.log.show_warning('rendered callback rasied error\n'
                                      + traceback.format_exc())
                self.remove_new_frame_callback(cb)

        return True

    def _block_redraw(self):
        # Avoid redrawing when we are already in the middle of drawing.
        self._block_redraw_count += 1
    def _unblock_redraw(self):
        self._block_redraw_count -= 1

    def report_framerate(self, time = None, monitor_period = 1.0):
        '''
        Report a status message giving the current rendering rate in frames per second.
        This is computed without the vertical sync which normally limits the frame rate
        to typically 60 frames per second.  The minimum drawing time used over a one
        second interval is used. The message is shown in the status line and log after
        the one second has elapsed.
        '''
        if time is None:
            from time import time
            self._time_graphics = time() + monitor_period
            self.minimum_render_time = None
            self.render_start_time = None
            self.redraw_needed = True
        else:
            self._time_graphics = 0
            msg = '%.1f frames/sec' % (1.0/time,)
            l = self.log
            l.show_status(msg)
            l.show_info(msg)

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
                self.report_framerate(mint)
            else:
                self.redraw_needed = True

    def finish_rendering(self):
        '''
        Force the graphics pipeline to complete all requested drawing.
        This can slow down rendering but is used by display devices such as Oculus Rift
        goggles to reduce latency time between head tracking and graphics update.
        '''
        self._render.finish_rendering()

    def add_new_frame_callback(self, cb):
        '''Add a function to be called before each redraw.  The function takes no arguments.'''
        self._new_frame_callbacks.append(cb)
    def remove_new_frame_callback(self, cb):
        '''Remove a callback that was added with add_new_frame_callback().'''
        self._new_frame_callbacks.remove(cb)

    def add_rendered_frame_callback(self, cb):
        '''Add a function to be called after each redraw.  The function takes no arguments.'''
        self._rendered_callbacks.append(cb)
    def remove_rendered_frame_callback(self, cb):
        '''Remove a callback that was added with add_rendered_frame_callback().'''
        self._rendered_callbacks.remove(cb)

    def add_shape_changed_callback(self, cb):
        '''
        Add a function to be called before each redraw when drawing shape has changed.
        The function takes no arguments.
        '''
        self._shape_changed_callbacks.append(cb)
    def remove_shape_changed_callback(self, cb):
        '''Remove a callback that was added with add_shape_changed_callback().'''
        self._shape_changed_callbacks.remove(cb)

    def _multishadow_directions(self):

        directions = self._multishadow_dir
        if directions is None or len(directions) != self.multishadow:
            n = self.multishadow    # requested number of directions
            from ..geometry import sphere
            self._multishadow_dir = directions = sphere.sphere_points(n)
        return directions

    def _draw_scene(self, camera = None, drawings = None):

        if camera is None:
            camera = self.camera

        mdraw = [self.drawing] if drawings is None else drawings

        r = self._render
        if self.shadows:
            kl = r.lighting.key_light_direction                     # Light direction in camera coords
            lightdir = camera.position.apply_without_translation(kl)  # Light direction in scene coords.
            stf = self._use_shadow_map(lightdir, drawings)
        if self.multishadow > 0:
            mstf, msdepth = self._use_multishadow_map(self._multishadow_directions(), drawings)

        r.set_background_color(self.background_color)

        if self.update_lighting:
            self.update_lighting = False
            r.set_shader_lighting_parameters()

        any_selected = self.any_drawing_selected() if drawings is None else True

        r.set_frame_number(self.frame_number)
        perspective_near_far_ratio = 2
        from .drawing import draw_depth, draw_drawings, draw_outline, draw_overlays
        for vnum in range(camera.number_of_views()):
            camera.set_render_target(vnum, r)
            if self.silhouettes:
                r.start_silhouette_drawing()
            r.draw_background()
            if mdraw:
                perspective_near_far_ratio = self._update_projection(vnum, camera = camera)
                cp = camera.get_position(vnum)
                cpinv = cp.inverse()
                if self.shadows:
                    r.set_shadow_transform(stf*cp)
                if self.multishadow > 0:
                    r.set_multishadow_transforms(mstf, cp, msdepth)
                    # Initial depth pass optimization to avoid lighting calculation on hidden geometry
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
                r.finish_silhouette_drawing(self.silhouette_thickness, self.silhouette_color,
                                            self.silhouette_depth_jump, perspective_near_far_ratio)

        camera.combine_rendered_camera_views(r)
        
        if self._overlays:
            draw_overlays(self._overlays, r)

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
        lvinv, stf = r.shadow_transforms(light_direction, center, radius)
        from .drawing import draw_drawings
        draw_drawings(r, lvinv, bdrawings)

        shadow_map = r.finish_rendering_shadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.shadow_texture_unit)

        return stf      # Scene to shadow map texture coordinates

    def _use_multishadow_map(self, light_directions, drawings):

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
        size = self.shadow_map_size
        r.start_rendering_multishadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # TODO: Don't run multishadow fragment shader when computing shadow map -- very expensive.
        mstf = []
        nl = len(light_directions)
        from .drawing import draw_drawings
        from math import ceil, sqrt
        d = int(ceil(sqrt(nl)))     # Number of subtextures along each axis
        s = size//d                 # Subtexture size.
        for l in range(nl):
            x, y = (l%d), (l//d)
            r.set_viewport(x*s, y*s, s, s)
            lvinv, tf = r.shadow_transforms(light_directions[l], center, radius)
            mstf.append(tf)
            draw_drawings(r, lvinv, bdrawings)

        shadow_map = r.finish_rendering_multishadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.multishadow_texture_unit)

        # TODO: Clear shadow cache whenever scene changes
        self._multishadow_transforms = mstf
        self._multishadow_depth = msd = 2*radius
#        r.set_multishadow_transforms(mstf, None, msd)
        return mstf, msd      # Scene to shadow map texture coordinates

    def drawing_bounds(self):
        '''Return bounds of drawing, displayed part only.'''
        dm = self._drawing_manager
        b = dm.cached_drawing_bounds
        if b is None:
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
        '''Set the camera position to show all displayed drawings, looking down the z axis.'''
        b = self.drawing_bounds()
        if b is None:
            return
        self.camera.initialize_view(b.center(), b.width())
        self.center_of_rotation = b.center()

    def view_all(self):
        '''Adjust the camera to show all displayed drawings using the current view direction.'''
        b = self.drawing_bounds()
        if b is None:
            return
        shift = self.camera.view_all(b.center(), b.width())
        self.translate(-shift)

    def center_of_rotation_needs_update(self):
        '''Cause the center of rotation to be updated.'''
        self._update_center = True

    def update_center_of_rotation(self):
        '''
        Update the center of rotation to the center of displayed drawings if zoomed out,
        or the front center object if zoomed in.
        '''
        if not self._update_center:
            return
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
                return
        self.center_of_rotation = cr

    def _front_center_point(self):
        w, h = self.window_size
        p, s = self.first_intercept(0.5*w, 0.5*h)
        return p

    def first_intercept(self, win_x, win_y):
        '''
        Return the position of the front-most object below the given screen window position (in pixels)
        and also return a Pick object describing the object.  This is used when hovering the mouse over
        an object (e.g. an atom) to get a description of that object.
        '''
        xyz1, xyz2 = self.clip_plane_points(win_x, win_y)
        f = None
        s = None
        drawings = self.drawing.child_drawings()
        for d in drawings:
            if d.display:
                fmin, smin = d.first_intercept(xyz1, xyz2, exclude = 'is_outline_box')
                if not fmin is None and (f is None or fmin < f):
                    f = fmin
                    s = smin
#        f, s = self.drawing.first_intercept(xyz1, xyz2, exclude = 'is_outline_box')
        if f is None:
            return None, None
        p = (1.0-f)*xyz1 + f*xyz2
        return p, s

    def _update_projection(self, view_num = None, camera = None):
        
        r = self._render
        ww,wh = r.render_size()
        if ww == 0 or wh == 0:
            return

        c = self.camera if camera is None else camera
        near,far = self._near_far_clip(c, view_num)
        pm = c.projection_matrix((near,far), view_num, (ww,wh))
        r.set_projection_matrix(pm)

        return near/far

    def _near_far_clip(self, camera, view_num):
        # Return the near and far clip plane distances.

        cp = camera.get_position(view_num).origin()
        vd = camera.view_direction(view_num)
        b = self.drawing_bounds()
        if b is None:
            return 0.001,1  # Nothing shown
        d = sum((b.center()-cp)*vd)         # camera to center of drawings
        w = b.width()
        near, far = (d - w, d + w)

        # Clamp near clip > 0.
        near_min = 0.001*(far - near) if far > near else 1
        near = max(near, near_min)
        if far <= near:
            far = 2*near

        return (near, far)

    def clip_plane_points(self, window_x, window_y, camera = None, view_num = None):
        '''
        Return two scene points at the near and far clip planes at the specified window pixel position.
        The points are in scene coordinates.
        '''
        c = camera if camera else self.camera
        nf = self._near_far_clip(c, view_num)
        scene_pts = c.clip_plane_points(window_x, window_y, self.window_size, nf, self._render)
        return scene_pts

    def rotate(self, axis, angle, drawings = None):
        '''
        Move camera to simulate a rotation of drawings about current rotation center.
        Axis is in scene coordinates and angle is in degrees.
        '''
        if drawings:
            from ..geometry import bounds
            b = bounds.union_bounds(d.bounds() for d in drawings)
            if b is None:
                return
            center = b.center()
        else:
            self.update_center_of_rotation()
            center = self.center_of_rotation
        from ..geometry import place
        r = place.rotation(axis, angle, center)
        self.move(r, drawings)

    def translate(self, shift, drawings = None):
        '''Move camera to simulate a translation of drawings.  Translation is in scene coordinates.'''
        self.center_of_rotation_needs_update()
        from ..geometry import place
        t = place.translation(shift)
        self.move(t, drawings)

    def move(self, tf, drawings = None):
        '''Move camera to simulate a motion of drawings.'''
        if drawings is None:
            c = self.camera
            c.position  = tf.inverse() * c.position
        else:
            for d in drawings:
                d.position = tf * d.position

        self.redraw_needed = True

    def pixel_size(self, p = None):
        '''Return the pixel size in scene length units at point p in the scene.'''
        if p is None:
            p = self.center_of_rotation
        return self.camera.pixel_size(p, self.window_size)

class OpenGLContext:
    '''
    OpenGL context used by View for drawing.
    This should be subclassed to provide window system specific opengl context methods.
    '''
    def make_current(self):
        '''Make the OpenGL context active.'''
        pass
    def swap_buffers(self):
        '''Swap back and front OpenGL buffers.'''
        pass

class _Redraw_Needed:
  def __init__(self):
    self.redraw_needed = False
    self.shape_changed = True
    self.cached_drawing_bounds = None
    self.cached_any_part_selected = None
  def __call__(self, shape_changed = False, selection_changed = False):
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
    radius = None if b is None else 0.5*b.width()
    return center, radius, bdrawings
