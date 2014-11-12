'''
View
====
'''

class View:
    '''
    A View is the graphics windows that shows 3-dimensional models.
    It manages the camera and draws the models when needed.
    '''
    def __init__(self, session, window_size, make_opengl_context_current, swap_opengl_buffers):

        self.session = session
        self.window_size = window_size		# pixels
        self._make_opengl_context_current = make_opengl_context_current
        self._swap_opengl_buffers = swap_opengl_buffers
        self._have_opengl_context = False

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

        self._overlays = []
        self.atoms_shown = 0

        from numpy import array, float32
        self.center_of_rotation = array((0,0,0), float32)
        self._update_center = True

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
        self._make_opengl_context_current()
        self._have_opengl_context = True
        self._initialize_opengl()

    def draw(self):
        '''Draw the scene.'''
        self._use_opengl()
        self._draw_scene()
        if self.camera.mode != 'oculus':
            self._swap_opengl_buffers()
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
        
    def set_camera_mode(self, mode):
        '''
        Camera mode can be 'mono', 'stereo' for sequential stereo, or
        'oculus' for side-by-side parallel view stereo used by Oculus Rift goggles.
        '''
        c = self.camera
        if mode == c.mode:
            return True

        if mode == 'stereo' or c.mode == 'stereo':
            if not self.enable_opengl_stereo(mode == 'stereo'):
                return False
        elif not mode in ('mono', 'oculus'):
            raise ValueError('Unknown camera mode %s' % mode)

        c.mode = mode
        self.redraw_needed = True

    def add_overlay(self, overlay):
        '''
        Overlays are Drawings rendered after the normal scene is shown.
        They are used for effects such as motion blur or cross fade that
        blend the current rendered scene with a previous rendered scene.
        '''
        overlay.redraw_needed = self.session.model_redraw_needed
        self._overlays.append(overlay)
        self.redraw_needed = True

    def overlays(self):
        '''The current list of overlay Drawings.''' 
        return self._overlays

    def remove_overlays(self, models = None):
        '''Remove the specified overlay Drawings.'''
        if models is None:
            models = self._overlays
        for o in models:
            o.delete()
        oset = set(models)
        self._overlays = [o for o in self._overlays if not o in oset]
        self.redraw_needed = True

    def image(self, width = None, height = None, supersample = None, camera = None, models = None):
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
            if models:
                from .camera import camera_framing_models
                c = camera_framing_models(models)
                if c is None:
                    return None         # Models not showing anything
            else:
                c = self.camera
        else:
            c = camera

        if supersample is None:
            self._draw_scene(c, models)
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
                    self._draw_scene(c, models)
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
        if self._have_opengl_context:
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
            return False

    def _draw_if_changed(self):
        for cb in self._new_frame_callbacks:
            try:
                cb()
            except:
                import traceback
                self.session.show_warning('new frame callback rasied error\n'
                                          + traceback.format_exc())
                self.remove_new_frame_callback(cb)
                

        c = self.camera
        s = self.session
        draw = self.redraw_needed or c.redraw_needed or s.redraw_needed
        if not draw:
            return False

        if s.redraw_needed and s.shape_changed and self.multishadow > 0:
            # Force recomputation of ambient shadows since shape changed.
            self._multishadow_transforms = []

        self.redraw_needed = False
        c.redraw_needed = False
        s.redraw_needed = False
        s.shape_changed = False
        self.draw()
        for cb in self._rendered_callbacks:
            try:
                cb()
            except:
                import traceback
                self.session.show_warning('rendered callback rasied error\n'
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
            s = self.session
            s.show_status(msg)
            s.show_info(msg)

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
        '''Add a callback that was added with add_new_frame_callback().'''
        self._new_frame_callbacks.remove(cb)

    def add_rendered_frame_callback(self, cb):
        '''Add a function to be called after each redraw.  The function takes no arguments.'''
        self._rendered_callbacks.append(cb)
    def remove_rendered_frame_callback(self, cb):
        '''Add a callback that was added with add_rendered_frame_callback().'''
        self._rendered_callbacks.remove(cb)

    def _multishadow_directions(self):

        directions = self._multishadow_dir
        if directions is None or len(directions) != self.multishadow:
            n = self.multishadow    # requested number of directions
            from ..geometry import sphere
            self._multishadow_dir = directions = sphere.sphere_points(n)
        return directions

    def _draw_scene(self, camera = None, models = None):

        if camera is None:
            camera = self.camera

        s = self.session
        mdraw = [m for m in s.top_level_models() if m.display] if models is None else models

        r = self._render
        if self.shadows:
            kl = r.lighting.key_light_direction                     # Light direction in camera coords
            lightdir = camera.view().apply_without_translation(kl)  # Light direction in scene coords.
            stf = self._use_shadow_map(lightdir, models)
        if self.multishadow > 0:
            mstf, msdepth = self._use_multishadow_map(self._multishadow_directions(), models)

        r.set_background_color(self.background_color)

        if self.update_lighting:
            self.update_lighting = False
            r.set_shader_lighting_parameters()

        self.update_level_of_detail()

        selected = [m for m in s.selected_models() if m.display]

        r.set_frame_number(self.frame_number)
        perspective_near_far_ratio = 2
        from .drawing import draw_depth, draw_drawings, draw_outline, draw_overlays
        for vnum in range(camera.number_of_views()):
            camera.set_framebuffer(vnum, r)
            if self.silhouettes:
                r.start_silhouette_drawing()
            r.draw_background()
            if mdraw:
                perspective_near_far_ratio = self._update_projection(vnum, camera = camera)
                if self.shadows:
                    r.set_shadow_transform(stf*camera.view())
                cvinv = camera.view_inverse(vnum)
                if self.multishadow > 0:
                    r.set_multishadow_transforms(mstf, camera.view(), msdepth)
                    # Initial depth pass optimization to avoid lighting calculation on hidden geometry
                    draw_depth(r, cvinv, mdraw)
                    r.allow_equal_depth(True)
                self._start_timing()
                draw_drawings(r, cvinv, mdraw)
                self._finish_timing()
                if self.multishadow > 0:
                    r.allow_equal_depth(False)
                if selected:
                    draw_outline(r, cvinv, selected)
            if self.silhouettes:
                r.finish_silhouette_drawing(self.silhouette_thickness, self.silhouette_color,
                                            self.silhouette_depth_jump, perspective_near_far_ratio)

        camera.combine_rendered_camera_views(r, self.session)
        
        if self._overlays:
            draw_overlays(self._overlays, r)

    def _use_shadow_map(self, light_direction, models):

        r = self._render

        # Compute model bounds so shadow map can cover all models.
        center, radius, models = _model_bounds(models, self.session)
        if center is None:
            return None

        # Compute shadow map depth texture
        size = self.shadow_map_size
        r.start_rendering_shadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # Compute light view and scene to shadow map transforms
        lvinv, stf = r.shadow_transforms(light_direction, center, radius)
        from .drawing import draw_drawings
        draw_drawings(r, lvinv, models)

        shadow_map = r.finish_rendering_shadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.shadow_texture_unit)

        return stf      # Scene to shadow map texture coordinates

    def _use_multishadow_map(self, light_directions, models):

        r = self._render
        if len(self._multishadow_transforms) == len(light_directions):
            # Bind shadow map for subsequent rendering of shadows.
            dt = r.multishadow_map_framebuffer.depth_texture
            dt.bind_texture(r.multishadow_texture_unit)
            return self._multishadow_transforms, self._multishadow_depth

        # Compute model bounds so shadow map can cover all models.
        center, radius, models = _model_bounds(models, self.session)
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
            draw_drawings(r, lvinv, models)

        shadow_map = r.finish_rendering_multishadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.multishadow_texture_unit)

        # TODO: Clear shadow cache whenever scene changes
        self._multishadow_transforms = mstf
        self._multishadow_depth = msd = 2*radius
#        r.set_multishadow_transforms(mstf, None, msd)
        return mstf, msd      # Scene to shadow map texture coordinates

    def update_level_of_detail(self):
        '''
        Update the level of detail used for rendering, for example,
        the number of triangles used to render spheres.
        '''
        # Level of detail updating.
        # TODO: Don't recompute atoms shown on every draw, only when changed
        ashow = sum(m.shown_atom_count() for m in self.session.molecules() if m.display)
        if ashow != self.atoms_shown:
            self.atoms_shown = ashow
            for m in self.session.molecules():
                m.update_level_of_detail(self)

    def initial_camera_view(self):
        '''Set the camera position to show all displayed models, looking down the z axis.'''
        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        from numpy import array, float32
        self.camera.initialize_view(center, s)
        self.center_of_rotation = center

    def view_all(self):
        '''Adjust the camera to show all displayed models using the current view direction.'''
        ses = self.session
        center, s = ses.bounds_center_and_width()
        if center is None:
            return
        shift = self.camera.view_all(center, s)
        self.translate(-shift)

    def center_of_rotation_needs_update(self):
        '''Cause the center of rotation to be updated.'''
        self._update_center = True

    def update_center_of_rotation(self):
        '''
        Update the center of rotation to the center of displayed models if zoomed out,
        or the front center object if zoomed in.
        '''
        if not self._update_center:
            return
        self._update_center = False
        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        vw = self.camera.view_width(center)
        if vw >= s:
            # Use center of models for zoomed out views
            cr = center
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
        models = self.session.top_level_models()
        for m in models:
            if m.display:
                fmin, smin = m.first_intercept(xyz1, xyz2, exclude = 'is_outline_box')
                if not fmin is None and (f is None or fmin < f):
                    f = fmin
                    s = smin
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

        cp = camera.position(view_num)
        vd = camera.view_direction(view_num)
        center, size = self.session.bounds_center_and_width()
        if center is None:
            return 0.001,1  # Nothing shown
        d = sum((center-cp)*vd)         # camera to center of models
        near, far = (d - size, d + size)

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

    def rotate(self, axis, angle, models = None):
        '''
        Move camera to simulate a rotation of models about current rotation center.
        Axis is in scene coordinates and angle is in degrees.
        '''
        if models:
            center = self.session.center(models)
        else:
            self.update_center_of_rotation()
            center = self.center_of_rotation
        from ..geometry import place
        r = place.rotation(axis, angle, center)
        self.move(r, models)

    def translate(self, shift, models = None):
        '''Move camera to simulate a translation of models.  Translation is in scene coordinates.'''
        self.center_of_rotation_needs_update()
        from ..geometry import place
        t = place.translation(shift)
        self.move(t, models)

    def move(self, tf, models = None):
        '''Move camera to simulate a motion of models.'''
        if models is None:
            c = self.camera
            cv = c.view()
            c.set_view(tf.inverse() * cv)
        else:
            for m in models:
                m.position = tf * m.position

        self.redraw_needed = True

    def pixel_size(self, p = None):
        '''Return the pixel size in scene length units at point p in the scene.'''
        if p is None:
            p = self.center_of_rotation
        return self.camera.pixel_size(p, self.window_size)

def _model_bounds(models, session):
    if models is None:
        s = session
        center, radius = s.bounds_center_and_width()
        models = [m for m in s.top_level_models() if m.display]
    else:
        from ..geometry import bounds
        b = bounds.union_bounds(m.bounds() for m in models)
        center, radius = bounds.bounds_center_and_radius(b)
    return center, radius, models
