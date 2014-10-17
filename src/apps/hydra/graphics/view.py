class View:
    '''
    A View is the graphics windows that shows 3-dimensional models.
    It manages the camera and draws the models when needed.
    '''
    def __init__(self, session, window_size):
        self.session = session

        self.window_size = window_size		# pixels
        self.background_rgba = (0,0,0,1)        # Red, green, blue, opacity, 0-1 range.

        # Create camera
        from ..graphics import Camera
        self.camera = Camera()
        '''The camera controlling the vantage shown in the graphics window.'''

        from . import Render
        self.render = Render()
        self.opengl_initialized = False
        self._shadows = False
        self.shadowMapSize = 2048
        self.multishadow = 0                    # Number of shadows
        self._multishadow_directions = None
        self._multishadow_transforms = []
        self._multishadow_depth = None
        self.silhouettes = False
        self.silhouette_thickness = 1           # pixels
        self.silhouette_color = (0,0,0,1)       # black
        self.silhouette_depth_jump = 0.01       # fraction of scene depth

        self.frame_number = 1
        self.redraw_needed = False
        self.update_lighting = False
        self.block_redraw_count = 0
        self.new_frame_callbacks = []
        self.rendered_callbacks = []
        self.last_draw_duration = 0             # seconds

        self.overlays = []
        self.atoms_shown = 0

        from numpy import array, float32
        self.center_of_rotation = array((0,0,0), float32)
        self.update_center = True

    def initialize_opengl(self):

        if self.opengl_initialized:
            return
        self.opengl_initialized = True

        r = self.render
        r.set_background_color(self.background_rgba)
        r.enable_depth_test(True)

        w,h = self.window_size
        r.initialize_opengl(w,h)

        from ..graphics import llgrutil as gr
        if gr.use_llgr:
            gr.initialize_llgr()

    def make_opengl_context_current(self):
        pass    # Defined by derived class
    def swap_opengl_buffers(self):
        pass    # Defined by derived class

    def use_opengl(self):
        self.make_opengl_context_current()
        self.initialize_opengl()

    def draw_graphics(self):
        self.use_opengl()
        self.draw_scene()
        self.swap_opengl_buffers()
        self.frame_number += 1

    def get_background_color(self):
        return self.background_rgba
    def set_background_color(self, rgba):
        self.background_rgba = tuple(rgba)
        self.redraw_needed = True
    background_color = property(get_background_color, set_background_color)

    def get_shadows(self):
        return self._shadows
    def set_shadows(self, onoff):
        onoff = bool(onoff)
        if onoff == self._shadows:
            return
        self._shadows = onoff
        r = self.render
        if onoff:
            r.enable_capabilities |= r.SHADER_SHADOWS
        else:
            r.enable_capabilities &= ~r.SHADER_SHADOWS
        self.redraw_needed = True
    shadows = property(get_shadows, set_shadows)

    def set_multishadow(self, n):
        self.multishadow = n
        r = self.render
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
        overlay.redraw_needed = self.session.model_redraw_needed
        self.overlays.append(overlay)
        self.redraw_needed = True

    def remove_overlays(self, models = None):
        if models is None:
            models = self.overlays
        for o in models:
            o.delete()
        oset = set(models)
        self.overlays = [o for o in self.overlays if not o in oset]
        self.redraw_needed = True

    def image(self, width = None, height = None, supersample = None, camera = None, models = None):

        self.use_opengl()

        w,h = self.window_size_matching_aspect(width, height)

        from .. import graphics
        fb = graphics.Framebuffer(w,h)
        if not fb.valid():
            return None         # Image size exceeds framebuffer limits

        r = self.render
        r.push_framebuffer(fb)
        c = camera if camera else self.camera
        if supersample is None:
            self.draw_scene(c, models)
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
                    self.draw_scene(c, models)
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

    def window_size_matching_aspect(self, width, height):
        w,h = width, height
        vw,vh = self.window_size
        if not w is None and not h is None:
            return (w,h)
        elif not w is None:
            return (w, (vh*w)//vw)     # Choose height to match window aspect ratio.
        elif not height is None:
            return ((vw*h)//vh, h)     # Choose width to match window aspect ratio.
        return (vw,vh)

    def renderer(self):
        return self.render

    def redraw(self):

        if self.block_redraw_count > 0:
            return False

        
        self.block_redraw()	# Avoid redrawing during callbacks of the current redraw.
        try:
            return self.redraw_graphics()
        finally:
            self.unblock_redraw()
            return False

    def redraw_graphics(self):
        for cb in self.new_frame_callbacks:
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
        self.draw_graphics()
        for cb in self.rendered_callbacks:
            try:
                cb()
            except:
                import traceback
                self.session.show_warning('rendered callback rasied error\n'
                                          + traceback.format_exc())
                self.remove_new_frame_callback(cb)

        return True

    def block_redraw(self):
        self.block_redraw_count += 1
    def unblock_redraw(self):
        self.block_redraw_count -= 1

    def add_new_frame_callback(self, cb):
        '''Add a function to be called before each redraw.  The function takes no arguments.'''
        self.new_frame_callbacks.append(cb)
    def remove_new_frame_callback(self, cb):
        '''Add a callback that was added with add_new_frame_callback().'''
        self.new_frame_callbacks.remove(cb)

    def add_rendered_frame_callback(self, cb):
        '''Add a function to be called after each redraw.  The function takes no arguments.'''
        self.rendered_callbacks.append(cb)
    def remove_rendered_frame_callback(self, cb):
        '''Add a callback that was added with add_rendered_frame_callback().'''
        self.rendered_callbacks.remove(cb)

    def multishadow_directions(self):

        directions = self._multishadow_directions
        if directions is None or len(directions) != self.multishadow:
            from ..surface import shapes
            n = self.multishadow    # requested number of directions
            from ..geometry import sphere
            self._multishadow_directions = directions = sphere.sphere_points(n)
        return directions

    def draw_scene(self, camera = None, models = None):

        if camera is None:
            camera = self.camera

        s = self.session
        mdraw = [m for m in s.top_level_models() if m.display] if models is None else models

        r = self.render
        if self.shadows:
            kl = r.lighting.key_light_direction                     # Light direction in camera coords
            lightdir = camera.view().apply_without_translation(kl)  # Light direction in scene coords.
            stf = self.use_shadow_map(lightdir, models)
        if self.multishadow > 0:
            mstf, msdepth = self.use_multishadow_map(self.multishadow_directions(), models)

        r.set_background_color(self.background_rgba)

        if self.update_lighting:
            self.update_lighting = False
            r.set_shader_lighting_parameters()

        self.update_level_of_detail()

        selected = [m for m in s.selected_models() if m.display]

        from time import time
        t0 = time()
        perspective_near_far_ratio = 2
        from .. import graphics
        for vnum in range(camera.number_of_views()):
            camera.set_framebuffer(vnum, r)
            if self.silhouettes:
                r.start_silhouette_drawing()
            r.draw_background()
            if mdraw:
                perspective_near_far_ratio = self.update_projection(vnum, camera = camera)
                if self.shadows:
                    r.set_shadow_transform(stf*camera.view())
                cvinv = camera.view_inverse(vnum)
                if self.multishadow > 0:
                    r.set_multishadow_transforms(mstf, camera.view(), msdepth)
                    # Initial depth pass optimization to avoid lighting calculation on hidden geometry
                    graphics.draw_depth(r, cvinv, mdraw)
                    r.allow_equal_depth(True)
                graphics.draw_drawings(r, cvinv, mdraw)
                if self.multishadow > 0:
                    r.allow_equal_depth(False)
                if selected:
                    graphics.draw_outline(r, cvinv, selected)
            if self.silhouettes:
                r.finish_silhouette_drawing(self.silhouette_thickness, self.silhouette_color,
                                            self.silhouette_depth_jump, perspective_near_far_ratio)
            s = camera.warp_image(vnum, r)
            if s:
                graphics.draw_overlays([s], r)

#        from OpenGL import GL
#        GL.glFinish()
        t1 = time()
        self.last_draw_duration = t1-t0

        
        if self.overlays:
            graphics.draw_overlays(self.overlays, r)

    def use_shadow_map(self, light_direction, models):

        r = self.render

        # Compute model bounds so shadow map can cover all models.
        center, radius, models = model_bounds(models, self.session)
        if center is None:
            return None

        # Compute shadow map depth texture
        size = self.shadowMapSize
        r.start_rendering_shadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # Compute light view and scene to shadow map transforms
        lvinv, stf = r.shadow_transforms(light_direction, center, radius)
        from .. import graphics
        graphics.draw_drawings(r, lvinv, models)

        shadow_map = r.finish_rendering_shadowmap()     # Depth texture

        # Bind shadow map for subsequent rendering of shadows.
        shadow_map.bind_texture(r.shadow_texture_unit)

        return stf      # Scene to shadow map texture coordinates

    def use_multishadow_map(self, light_directions, models):

        r = self.render
        if len(self._multishadow_transforms) == len(light_directions):
            # Bind shadow map for subsequent rendering of shadows.
            dt = r.multishadow_map_framebuffer.depth_texture
            dt.bind_texture(r.multishadow_texture_unit)
            return self._multishadow_transforms, self._multishadow_depth

        # Compute model bounds so shadow map can cover all models.
        center, radius, models = model_bounds(models, self.session)
        if center is None:
            return None, None

        # Compute shadow map depth texture
        size = self.shadowMapSize
        r.start_rendering_multishadowmap(center, radius, size)
        r.draw_background()             # Clear shadow depth buffer

        # TODO: Don't run multishadow fragment shader when computing shadow map -- very expensive.
        mstf = []
        nl = len(light_directions)
        from ..graphics import draw_drawings
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
        # Level of detail updating.
        # TODO: Don't recompute atoms shown on every draw, only when changed
        ashow = sum(m.shown_atom_count() for m in self.session.molecules() if m.display)
        if ashow != self.atoms_shown:
            self.atoms_shown = ashow
            for m in self.session.molecules():
                m.update_level_of_detail(self)

    def initial_camera_view(self):

        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        from numpy import array, float32
        self.camera.initialize_view(center, s)
        self.center_of_rotation = center

    def view_all(self):
        '''Adjust the camera to show all displayed models.'''
        ses = self.session
#        ses.bounds_changed = True       # TODO: Model display changes are not setting bounds changed flag.
        center, s = ses.bounds_center_and_width()
        if center is None:
            return
        shift = self.camera.view_all(center, s)
        self.translate(-shift)

    def center_of_rotation_needs_update(self):
        self.update_center = True

    def update_center_of_rotation(self):
        if not self.update_center:
            return
        self.update_center = False
        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        vw = self.camera.view_width(center)
        if vw >= s:
            # Use center of models for zoomed out views
            cr = center
        else:
            # Use front center point for zoomed in views
            cr = self.front_center_point()
            if cr is None:
                return
        self.center_of_rotation = cr

    def front_center_point(self):
        w, h = self.window_size
        p, s = self.first_intercept(0.5*w, 0.5*h)
        return p

    def first_intercept(self, win_x, win_y):
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

    def update_projection(self, view_num = None, camera = None):
        
        r = self.render
        ww,wh = r.render_size()
        if ww == 0 or wh == 0:
            return

        c = self.camera if camera is None else camera
        near,far = self.near_far_clip(c, view_num)
        pm = c.projection_matrix((near,far), view_num, (ww,wh))
        r.set_projection_matrix(pm)

        return near/far

    def near_far_clip(self, camera, view_num):

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
        Two scene points at the near and far clip planes at the specified window pixel position.
        The points are in scene coordinates.
        '''
        c = camera if camera else self.camera
        nf = self.near_far_clip(c, view_num)
        scene_pts = c.clip_plane_points(window_x, window_y, self.window_size, nf, self.render)
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

def model_bounds(models, session):
    if models is None:
        s = session
        center, radius = s.bounds_center_and_width()
        models = [m for m in s.top_level_models() if m.display]
    else:
        from ..geometry import bounds
        b = bounds.union_bounds(m.bounds() for m in models)
        center, radius = bounds.bounds_center_and_radius(b)
    return center, radius, models
