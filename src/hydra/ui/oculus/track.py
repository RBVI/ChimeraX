# -----------------------------------------------------------------------------
#
class Oculus_Head_Tracking:

    def __init__(self):

        self.connected = False
        self.last_rotation = None
        self.view = None
        self.lens_correction = True
        self.scale = 1.5 	       # Scale eye images to use edges of oculus display.
        self.last_angle = 0
        self.last_axis = (0,0,1)
        self.min_angle_change = 1e-4
        self.min_axis_change = 1e-4
        self.predict_orientation = 0.030        # Time (seconds) in future to predict oculus orientation.
        self.frame_cb = None

    def start_event_processing(self, view):

        from . import _oculus

        if not self.connected:
            try:
                _oculus.connect()
                self.connected = True
            except:
                return False

        _oculus.set_prediction_time(self.predict_orientation)

        self.parameters = p = _oculus.parameters()
        for k,v in p.items():
            print (k,v)
        from math import pi
        print('oculus field of view %.1f degrees' % (self.field_of_view()*180/pi))
        print('oculus image shift %.1f pixels' % self.image_shift_pixels())

        self.view = view
        if self.frame_cb is None:
            self.frame_cb = self.use_oculus_orientation
            view.add_new_frame_callback(self.frame_cb)
        return True

    def stop_event_processing(self):
        if self.frame_cb:
            self.view.remove_new_frame_callback(self.frame_cb)
            self.frame_cb = None

    def display_size(self):

        p = self.parameters
        w, h = int(p['HResolution']), int(p['VResolution'])     # SDK values 1280, 800
        return w,h

    def field_of_view(self):

        # Horizontal field of view from Oculus SDK Overview.
        p = self.parameters
        d = self.eye_to_screen_distance()
        w = p['HScreenSize']            # meters, devkit value 0.15
        s = self.scale
        from math import atan2
        fov = 2*atan2(0.25*w*s, d)
        return fov

    def eye_to_screen_distance(self):
        #
        # Using the eye to screen distance reported by the developer kit of 41 mm gives warping
        # as I view side to side as if the field of view is wrong.  I tried different field of view
        # values and found that 0.90 times the calculated value gave much less warping.
        # The eye to screen distance for field of view calculation should be the screen to the center
        # of lens distance.  Direct measurements show that number is 47 mm for the "A" lenses (no
        # diopter correction), 44 mm for the B lenses and 40 mm for the C lenses.  With 47 mm the
        # warping is much reduced and this corresponds to 0.90 times field of view with 41 mm
        # (85 degree horz field with 41 mm, and 77 degree horz field with 47 mm).  Testing with
        # the C lenses 41 mm worked well with little warp while 47 mm showed significant warping.
        # Tests with the oculus configuration tool where the lenses A, B, C can be specified show
        # it does not change the eye distance 41 mm reported by the SDK.  Posts on the web also
        # show that the 41 mm is hard-coded in the SDK independent of lenses.
        #
        # TODO: For now I override the SDK value to give minimal warping.
        #
        lens = 'A'
        if lens == 'A':
            d = 0.047
        elif lens == 'B':
            d = 0.044
        elif lens == 'C':
            d = 0.040
        else:
            d = self.parameters['EyeToScreenDistance']	# meters, devkit value 0.041
        return d

    def image_shift_pixels(self):

        # Projection center shift from middle of viewport for each eye.
        # Taken from Oculus SDK Overview, page 24
        # Center should be at lens separation / 2 instead of viewport width / 2.
        p = self.parameters
        w = p['HScreenSize']                    # meters, devkit value 0.15
        hw = 0.5*w
        s = p['LensSeparationDistance']         # meters, devkit value 0.0635, InterpupillaryDistance 0.064
        dx = 0.5*s - 0.5*hw                     # meters
        wpx = p['HResolution']                  # pixels, devkit value 1280
        ppm = wpx / w       # pixels per meter
        xp = dx * ppm
        return xp

    def radial_warp_parameters(self):

        # TODO:
        # Experiments show that increasing the space between the human eye and lense
        # using the face mask adjustment causes the radial warping correction to not
        # work as well.  The SDK parameters seem to work well with the eyes close to
        # the lenses ("A" lenses, no diopter correction).  But if the maximum spacing
        # between eyes and lenses is used, with or without glasses, there is very
        # noticable warping where an object crossing the field of view appears to change
        # its distance, far then near then far, with the near position in center of view.
        # This in and out motion is somewhat nauseating.
        p = self.parameters
        k0,k1,k2,k3 =  p['DistortionK']         # devkit values (1.0, 0.22, 0.24, 0)
        s = self.scale
        return (k0/s, k1/s, k2/s, k3/s)

    def chromatic_aberration_parameters(self):
        p = self.parameters
        cab = p['ChromaAbCorrection']          # defkit values (0.996, -0.004, 1.014, 0)
        return cab

    def use_oculus_orientation(self):

        self.view.render.finish_rendering()     # Reduce latency by finishing current graphics draw.

        from . import _oculus
        q = _oculus.state()
        w,x,y,z = q     # quaternion orientation = (cos(a/2), axis * sin(a/2))

        from math import sqrt, atan2, pi
        vn = sqrt(x*x+y*y+z*z)
        a = 2*atan2(vn, w)
        axis = (x/vn, y/vn, z/vn) if vn > 0 else (0,0,1)
        if (abs(a - self.last_angle) < self.min_angle_change or
            max(abs(x1-x0) for x0, x1 in zip(axis, self.last_axis)) < self.min_axis_change):
            return
        self.last_angle, self.last_axis = a, axis
        from ...geometry import place
        r = place.rotation(axis, a*180/pi)
        if not self.last_rotation is None:
            rdelta = self.last_rotation.inverse()*r
            v = self.view
            c = v.camera
            mtf = c.view()*rdelta.inverse()*c.view_inverse()
            v.move(mtf)
        self.last_rotation = r

    def set_camera_mode(self, view):
        if self.connected:
            fov = self.field_of_view()
            ishift = self.image_shift_pixels()
            warp = self.radial_warp_parameters()
            cwarp = self.chromatic_aberration_parameters()
            w,h = self.display_size()
            s = self.scale
            wsize = (int(s*0.5*w), int(s*h))
        else:
            from math import atan2
            fov = 1.75              # 100 degrees, scale 1.5 value.
            ishift = -49
            s = self.scale
            warp = (1.0/s, 0.22/s, 0.24/s, 0/s)
            cwarp = (0.996, -0.004, 1.014, 0)
            w,h = (1280,800)
            wsize = (int(s*0.5*w), int(s*h))

        c = view.camera
        from math import pi
        c.field_of_view = fov * 180 / pi
        c.eye_separation_scene = 0.2        # TODO: This is good value for inside a molecule, not for far from molecule.
        c.eye_separation_pixels = 2*ishift
        view.set_camera_mode('oculus')
        r = view.render
        if self.lens_correction:
            c.warp_window_size = wsize
            r.radial_warp_coefficients = warp
            r.chromatic_warp_coefficients = cwarp
        else:
            c.warp_window_size = (w//2,h)
            r.radial_warp_coefficients = (1,0,0,0)
            r.chromatic_warp_coefficients = (1,0,1,0)

        view.redraw_needed = True

def start_oculus(session):

    oht = session.oculus
    if oht is None:
        session.oculus = oht = Oculus_Head_Tracking()

    success = oht.start_event_processing(session.view)
    msg = 'started oculus head tracking ' if success else 'failed to start oculus head tracking'
    session.show_status(msg)
    session.show_info(msg)
    if oht.connected:
        oculus_full_screen(True, session)

    oht.set_camera_mode(session.view)

def oculus_on(session):

    oht = session.oculus
    return oht and oht.connected and oht.frame_cb

def stop_oculus(session):

    oht = session.oculus
    if oht is None:
        return
    oht.stop_event_processing()

    oculus_full_screen(False, session)

    v = session.view
    v.set_camera_mode('mono')
    v.camera.field_of_view = 30.0

# Go full screen on oculus display
def oculus_full_screen(full, session):
    d = session.application.desktop()
    mw = session.main_window
    if full and mw.toolbar.isVisible():
        mw.toolbar.hide()
        mw.show_command_line(False)
        mw.statusBar().hide()
        if session.oculus.connected:
            w,h = session.oculus.display_size()
            move_window_to_oculus(session, w, h)
    else:
        move_window_to_primary_screen(session)
        mw.toolbar.show()
        mw.show_command_line(True)
        mw.statusBar().show()

def toggle_warping(session):
    r = session.view.render
    oht = session.oculus
    if r.radial_warp_coefficients == (1,0,0,0):
        if oht.connected:
            wc = oht.radial_warp_parameters()
            cwc = oht.chromatic_aberration_parameters()
        else:
            s = oht.scale
            wc = (1.0/s, 0.22/s, 0.24/s, 0/s)
            cwc = (0.996, -0.004, 1.014, 0)
    else:
        wc = (1,0,0,0)
        cwc = (1,0,1,0)

    r.radial_warp_coefficients = wc
    r.chromatic_warp_coefficients = cwc
    session.view.redraw_needed = True

def move_window_to_oculus(session, w, h):
    d = session.application.desktop()
    mw = session.main_window
    for s in range(d.screenCount()):
        g = d.screenGeometry(s)
        if g.width() == w and g.height() == h:
            mw.move(g.left(), g.top())
            break
    mw.resize(w,h)

# TODO: Unfortunately a full screen app on a second display blanks the primary display in Mac OS 10.8.
# I believe this is fixed in Mac OS 10.9.
#    mw.showFullScreen()

def move_window_to_primary_screen(session):
    d = session.application.desktop()
    s = d.primaryScreen()
    g = d.screenGeometry(s)
    mw = session.main_window
    x,y = (g.width() - mw.width())//2, (g.height() - mw.height())//2
    mw.move(x, 0)       # Work around bug where y-placement is wrong.
    mw.move(x, y)

def oculus_command(enable = None, scale = None, lensCorrection = None, session = None):

    if not enable is None:
        if enable:
            start_oculus(session)
        else:
            stop_oculus(session)

    if not scale is None:
        oht = session.oculus
        if oht:
            oht.scale = scale
            oht.set_camera_mode(session.view)

    if not lensCorrection is None:
        oht = session.oculus
        if oht:
            oht.lens_correction = lensCorrection
            oht.set_camera_mode(session.view)
