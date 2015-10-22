# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
#
class Oculus_Rift:

    def __init__(self, session):

        self.view = session.main_view
        self.ses_triggers = session.triggers

        self.last_translation = None
        self.last_rotation = None
        self.panning_speed = 5
        self.frame_cb = None

        self.connected = self.connect()

    def connect(self):
        from . import _oculus
        try:
            _oculus.connect()
            c = True
        except:
            c = False

        if c:
            self.parameters = p = _oculus.parameters()
            params = list(p.items())
            params.sort()
            for k,v in params:
                print (k,v)
            print('oculus field of view %.1f degrees' % (self.field_of_view_degrees()))
            print('oculus camera centering shift %.1f, %.1f pixels, left eye' % self.camera_centering_shift_pixels())

        return c

    def start_event_processing(self):

        if self.frame_cb is None:
            from . import _oculus
            _oculus.initialize()
            self.parameters = _oculus.parameters()      # Interpupillary distance not set until initialize called.
            self.frame_cb = self.use_oculus_orientation
            self.handler = self.ses_triggers.add_handler('new frame', self.frame_cb)
        return True

    def close(self):

        self.stop_event_processing()

        from ...graphics import MonoCamera
        self.view.camera = MonoCamera()

    def stop_event_processing(self):

        if self.frame_cb:
            self.ses_triggers.delete_handler(self.handler)
            self.frame_cb = None
            if self.connected:
                from . import _oculus
                _oculus.disconnect()
                self.connected = False

    def display_size(self):

        p = self.parameters
        w, h = p['width'], p['height']     # (1280,800) for DK1, (1920,1080) for DK2
        return w,h

    def eye_render_size(self):

        p = self.parameters
        w, h = p['texture width'], p['texture height']  # (1182,1461) for DK2
        return w,h

    def render(self, tex_width, tex_height, tex_left, tex_right):

        if self.connected and self.frame_cb:
            from . import _oculus
            _oculus.render(tex_width, tex_height, tex_left, tex_right)

    def field_of_view_degrees(self):

        # Horizontal field of view
        p = self.parameters
        from math import atan, pi
        a = atan(p['fov left']) + atan(p['fov right'])
        fov = a*180/pi  # 94.2 degrees for DK2
        return fov

    def camera_centering_shift_pixels(self):

        # Projection center shift in texture pixels from middle of viewport for left eye.
        p = self.parameters
        w,h = p['texture width'], p['texture height']          # pixels, 1182 x 1461 for DK2
        lt = p['fov left']              # tangent left left angle, for left eye, 1.058 for DK2
        rt = p['fov right']             # tangent right half angle, for left eye, 1.092 for DK2
        ut = p['fov up']                # tangent up half angle, for left eye, 1.329 for DK2
        dt = p['fov down']              # tangent down half angle, for left eye, 1.329 for DK2
        sx, sy = w * (lt/(rt+lt) - 0.5), h * (dt/(ut+dt) - 0.5)
        return sx, sy

    def use_oculus_orientation(self, *_):

        v = self.view
        c = v.camera
        v.finish_rendering()     # Reduce latency by finishing current graphics draw.

        from . import _oculus
        x,y,z,qw,qx,qy,qz = _oculus.state()

        from ...geometry import place
        if qw is None:
            r = None
        else:
            from math import sqrt, atan2, pi
            vn = sqrt(qx*qx+qy*qy+qz*qz)
            a = 2*atan2(vn, qw)
            axis = (qx/vn, qy/vn, qz/vn) if vn > 0 else (0,0,1)
            r = place.rotation(axis, a*180/pi)

        if x is None:
            t = None
        else:
            ipd = self.parameters['interpupillary distance']
            s = self.panning_speed * c.eye_separation_scene / ipd
            t = place.translation((s*x,s*y,s*z))

        mdelta = self.relative_motion(t,r)
        cp = c.position
        mtf = cp*mdelta*cp.inverse()
        v.move(mtf)

        self.last_translation = t
        self.last_rotation = r

    def relative_motion(self, t, r):
        lt, lr = self.last_translation, self.last_rotation
        no_t = (t is None or lt is None)
        no_r = (r is None or lr is None)
        if no_t:
            if no_r:
                from ...geometry import place
                rel = place.Place()
            else:
                rel = r.inverse()*lr
        elif no_r:
            t.inverse()*lt
        else:
            lm = lt*lr
            m = t*r
            rel = m.inverse()*lm
        return rel

from ...graphics import Camera
class Oculus_Rift_Camera(Camera):

    def __init__(self, oculus_rift, oculus_opengl_context, render_opengl_context):

        Camera.__init__(self)

        self.oculus_rift = oculus_rift

        self.oculus_opengl_context = oculus_opengl_context
        self.render_opengl_context = render_opengl_context

        oc = oculus_rift
        fov = oc.field_of_view_degrees()
        sx,sy = oc.camera_centering_shift_pixels()
        w,h = oc.display_size()
        wsize = oc.eye_render_size()

        self.field_of_view = fov

        self.eye_separation_scene = 0.2    # TODO: This is good value for inside a molecule, not for far from molecule.
        "Stereo eye separation in scene units."

        self.oculus_centering_shift = (sx,sy)
        '''
        For the oculus rift the camera is not centered in the window.
        This parameter gives the x and y pixel shifts from the geometric center for the left eye.
        This should be removed from Camera and handled by a generic method the sets the render target.
        '''

        self._warp_framebuffers = [None, None]   # Off-screen rendering each eye for Oculus Rift
        self.warp_window_size = wsize
        '''
        Texture render size for each eye when using Oculus Rift.
        This should be removed from Camera and handled by a generic method the sets the render target.
        '''

        self.debug_oculus = False      # Render unwarped textures instead of having Oculus SDK do it.

        # Used mostly for demoing oculus eye views can be seen on conventional screen
        self.draw_main_window = (oculus_rift is None)

    def name(self):
        '''Name of camera.'''
        return 'oculus'

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            s = -1 if view_num == 0 else 1
            es = self.eye_separation_scene
            from ...geometry import place
            t = place.translation((s*0.5*es,0,0))
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        return 2

    def view_width(self, point):
        from ...graphics.camera import perspective_view_width
        return perspective_view_width(point, self.position.origin(), self.field_of_view)

    def pixel_shift(self, view_num):
        '''Shift of center away from center of render target.'''
        if view_num is None:
            return 0,0
        s = 1 if view_num == 0 else -1
        sx,sy = self.oculus_centering_shift # For left eye
        return (s*sx, s*sy)

    def view_all(self, center, size):
        from ...graphics import camera
        self.position = camera.perspective_view_all(center, size, self.position, self.field_of_view)

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        if view_num > 0:
            render.pop_framebuffer()
        render.push_framebuffer(self._warping_framebuffer(view_num))

    def combine_rendered_camera_views(self, render):
        '''Combine left and right eye images from separate textures with warping for Oculus Rift.'''

        render.pop_framebuffer()

        if self.draw_main_window:
            # Draw in main window.
            # This causes unpleasant oculus frame drop because vsync on main window is slower than oculus.
            fb = render.current_framebuffer()
            self._draw_unwarped(render, fb.width, fb.height)

        # Draw in oculus window
        o = self.oculus_rift
        if not o is None:
            if self.debug_oculus:
                self._draw_oculus_unwarped(render)
            else:
                t0,t1 = [rb.color_texture for rb in self._warp_framebuffers]
                self.oculus_opengl_context.make_current()
                o.render(t0.size[0], t0.size[1], t0.id, t1.id)
                # Switch back to main window graphics context.
                self.render_opengl_context.make_current()
                render.opengl_context_changed()

    def do_swap_buffers(self):
        return self.oculus_rift is None

    def _warping_framebuffer(self, view_num):

        tw,th = self.warp_window_size
        
        fb = self._warp_framebuffers[view_num]
        if fb is None or fb.width != tw or fb.height != th:
            from ...graphics import Texture, opengl
            t = Texture()
            t.initialize_rgba((tw,th))
            self._warp_framebuffers[view_num] = fb = opengl.Framebuffer(color_texture = t)
        return fb

    def _draw_oculus_unwarped(self, render):

        # Unwarped rendering, for testing purposes.
        self.oculus_opengl_context.make_current()
        render.opengl_context_changed()

        w,h = self.oculus_rift.display_size()
        self._draw_unwarped(render, w, h)

        self.oculus_opengl_context.swap_buffers()

        self.render_opengl_context.make_current()
        render.opengl_context_changed()

    def _draw_unwarped(self, render, w, h):

        # Draw left eye
        render.set_viewport(0,0,w//2,h)

        s = self._warping_surface(render)
        s.texture = self._warp_framebuffers[0].color_texture
        from ...graphics.drawing import draw_overlays
        draw_overlays([s], render)

        # Draw right eye
        render.set_viewport(w//2,0,w//2,h)
        s.texture = self._warp_framebuffers[1].color_texture
        draw_overlays([s], render)

    def _warping_surface(self, render):

        if hasattr(self, '_warp_surface'):
            return self._warp_surface

        from ...graphics import Drawing
        self._warp_surface = s = Drawing('warp plane')
        # TODO: Use a childless drawing.
        from numpy import array, float32, int32
        va = array(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)), float32)
        ta = array(((0,1,2),(0,2,3)), int32)
        tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
        s.geometry = va, ta
        s.color = (255,255,255,255)
        s.use_lighting = False
        s.texture_coordinates = tc
        return s
