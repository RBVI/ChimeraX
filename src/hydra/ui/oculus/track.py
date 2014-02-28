# -----------------------------------------------------------------------------
#
class Oculus_Head_Tracking:

    def __init__(self):

        self.last_rotation = None
        self.view = None
        self.scale = 1.0        # Scale eye images to fill oculus display.
        self.last_angle = 0
        self.last_axis = (0,0,1)
        self.min_angle_change = 1e-4
        self.min_axis_change = 1e-4
        self.predict_orientation = 0.030        # Time (seconds) in future to predict oculus orientation.

    def start_event_processing(self, view):

        from . import _oculus
        try:
            _oculus.connect()
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
        view.add_new_frame_callback(self.use_oculus_orientation)
        return True

    def display_size(self):

        p = self.parameters
        w, h = int(p['HResolution']), int(p['VResolution'])     # SDK values 1280, 800
        return w,h

    def field_of_view(self):

        # Horizontal field of view from Oculus SDK Overview.
        # For oculus developer kit, EyeToScreenDistance is 4.1 cm and HScreenSize is 15 cm.
        # The actual distance from center of lens to screen at closest accordion setting is 5 cm.
        # So it appears EyeToScreenDistance is an effective value accounting for magnification.
        p = self.parameters
        d = p['EyeToScreenDistance']	# meters, devkit value 0.041
        w = p['HScreenSize']            # meters, devkit value 0.15
        s = self.scale
        from math import atan2
        fov = 2*atan2(0.25*w*s, d)
# TODO: Using 90% of the field of view seems to give less warping when rotating head.
#       Maybe I am not computing the field of view correctly.  Do I need to apply a spherical
#       aberration correction?  Not according to SDK docs. But SDK docs describe vertical field
#       of view and I am using horizontal field of view.  Aha!  But eye is not centered in horz
#       half display.  But my rendering is so it should be fine.  Needs more study.
#        fov = 0.9*2*atan2(0.25*w*s, d)
        return fov

    def image_shift_pixels(self):

        # Projection center shift from middle of viewport for each eye.
        # Taken from Oculus SDK Overview, page 24
        # Center should be at lens separation / 2 instead of viewport width / 2.
        p = self.parameters
        w = p['HScreenSize']                    # meters, devkit value 0.15, VScreenSize 0.0936
        hw = 0.5*w
        s = p['LensSeparationDistance']         # meters, devkit value 0.0635, InterpupillaryDistance 0.064
        dx = 0.5*s - 0.5*hw                     # meters
        wpx = p['HResolution']                  # pixels, devkit value 1280
        ppm = wpx / w       # pixels per meter
        xp = dx * ppm
        return xp

    def radial_warp_parameters(self):

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
            v.move(mtf, update_clip_planes = True)
        self.last_rotation = r

def start_oculus(session):

    oht = session.oculus
    start = (oht is None)
    if start:
        session.oculus = oht = initialize_head_tracking(session)

    set_oculus_camera_mode(session)

    if oht:
        oculus_full_screen(start, session)

def initialize_head_tracking(session):

    oht = Oculus_Head_Tracking()
    view = session.view
    success = oht.start_event_processing(view)
    msg = 'started oculus head tracking ' + ('success' if success else 'failed')
    session.show_status(msg)
    session.show_info(msg)
    return oht if success else None

# Set stereo camera mode
def set_oculus_camera_mode(session):
    oht = session.oculus
    if oht:
        fov = oht.field_of_view()
        ishift = oht.image_shift_pixels()
        warp = oht.radial_warp_parameters()
        cwarp = oht.chromatic_aberration_parameters()
        print ('Radial warp', warp)
    else:
        fov = 1.5
        ishift = -50
        warp = (1, 0, 0, 0)
        cwarp = (1, 0, 1, 0)

    view = session.view
    c = view.camera
    from math import pi
    c.field_of_view = fov * 180 / pi
    c.eye_separation_scene = 0.2        # TODO: This is good value for inside a molecule, not for far from molecule.
    c.eye_separation_pixels = 2*ishift
    view.set_camera_mode('oculus')
    r = view.render
    r.radial_warp_coefficients = warp
    r.chromatic_warp_coefficients = cwarp

# Go full screen on oculus display
def oculus_full_screen(full, session):
    d = session.application.desktop()
    mw = session.main_window
    if full or mw.toolbar.isVisible():
        mw.toolbar.hide()
        mw.command_line.hide()
        mw.statusBar().hide()
        w,h = session.oculus.display_size()
        move_window_to_oculus(session, w, h)
    else:
        move_window_to_primary_screen(session)
        mw.toolbar.show()
        mw.command_line.show()
        mw.statusBar().show()

def toggle_warping(session):
    r = session.view.render
    if r.radial_warp_coefficients == (1,0,0,0):
        r.radial_warp_coefficients = session.oculus.radial_warp_parameters()
    else:
        r.radial_warp_coefficients = (1,0,0,0)

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

