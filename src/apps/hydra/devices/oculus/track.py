# -----------------------------------------------------------------------------
#
class Oculus_Rift:

    def __init__(self, session):

        self.session = session
        self.connected = False
        self.view = None

        self.last_translation = None
        self.last_rotation = None
        self.panning_speed = 5
        self.frame_cb = None

        # Create separate graphics window for rendering to Oculus Rift.
        from ...ui.qt import graphicswindow as gw
        self.window = gw.Secondary_Graphics_Window('Oculus Rift View', session)

        success = self.start_event_processing(session.view)
        msg = 'started oculus head tracking ' if success else 'failed to start oculus head tracking'
        session.show_status(msg)
        session.show_info(msg)

        if success:
            self.oculus_full_screen()
            self.set_camera_mode(session.view)

    def close(self):

        self.stop_event_processing()
        self.window.close()
        self.window = None

        v = self.session.view
        v.set_camera_mode('mono')
        v.camera.field_of_view = 30.0

    def start_event_processing(self, view):

        from . import _oculus

        if not self.connected:
            from ...graphics import opengl
            b = opengl.deactivate_bindings() # Make sure oculus doesn't change current VAO.
            try:
                _oculus.connect()
                self.connected = True
            except:
                return False

        self.parameters = p = _oculus.parameters()
        params = list(p.items())
        params.sort()
        for k,v in params:
            print (k,v)
        print('oculus field of view %.1f degrees' % (self.field_of_view_degrees()))
        print('oculus camera centering shift %.1f, %.1f pixels, left eye' % self.camera_centering_shift_pixels())

        self.view = view
        if self.frame_cb is None:
            self.frame_cb = self.use_oculus_orientation
            view.add_new_frame_callback(self.frame_cb)
        return True

    def stop_event_processing(self):

        if self.frame_cb:
            self.view.remove_new_frame_callback(self.frame_cb)
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

    # Move window to oculus screen and switch to full screen mode.
    def oculus_full_screen(self):

        if not self.connected:
            return

        w,h = self.display_size()
        self.window.full_screen(w,h)

    def render(self, tex_width, tex_height, tex_left, tex_right):

        if self.connected and self.frame_cb:
            self.window.make_opengl_context_current()
            from . import _oculus
            _oculus.render(tex_width, tex_height, tex_left, tex_right)
            self.session.view.make_opengl_context_current()

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

    def use_oculus_orientation(self):

        v = self.view
        c = v.camera
        v.render.finish_rendering()     # Reduce latency by finishing current graphics draw.

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
            s = self.panning_speed * c.eye_separation_scene / self.parameters['interpupillary distance']
            t = place.translation((s*x,s*y,s*z))

        mdelta = self.relative_motion(t,r)
        mtf = c.view()*mdelta*c.view_inverse()
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

    def set_camera_mode(self, view):
        if not self.connected:
            return

        fov = self.field_of_view_degrees()
        sx,sy = self.camera_centering_shift_pixels()
        w,h = self.display_size()
        wsize = self.eye_render_size()

        c = view.camera
        from math import pi
        c.field_of_view = fov
        print ('set camera field of view', fov)
        c.eye_separation_scene = 0.2        # TODO: This is good value for inside a molecule, not for far from molecule.
        c.oculus_centering_shift = (sx,sy)
        print ('oculus camera shift pixels', sx, sy)
        view.set_camera_mode('oculus')
        r = view.render
        c.warp_window_size = wsize

        view.redraw_needed = True

def start_oculus(session):

    oc = session.oculus
    if oc is None:
        session.oculus = oc = Oculus_Rift(session)

def stop_oculus(session):

    oc = session.oculus
    if oc:
        oc.close()
        session.oculus = None

def oculus_command(enable = None, panSpeed = None, session = None):

    if not enable is None:
        if enable:
            start_oculus(session)
        else:
            stop_oculus(session)

    if not panSpeed is None:
        oc = session.oculus
        if oc:
            oc.panning_speed = panSpeed
