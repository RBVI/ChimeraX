# -----------------------------------------------------------------------------
#
class Oculus_Head_Tracking:

    def __init__(self):

        self.last_rotation = None
        self.view = None

    def start_event_processing(self, view):

        from . import _oculus
        try:
            _oculus.connect()
        except:
            return False

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
        w, h = int(p['HResolution']), int(p['VResolution'])
        return w,h

    def field_of_view(self):

        # Horizontal field of view from Oculus SDK Overview.
        # For oculus developer kit, EyeToScreenDistance is 4.1 cm and HScreenSize is 15 cm.
        # The actual distance from center of lens to screen at closest accordion setting is 5 cm.
        # So it appears EyeToScreenDistance is an effective value accounting for magnification.
        p = self.parameters
        from math import atan2
        fov = 2*atan2(0.25*p['HScreenSize'], p['EyeToScreenDistance'])
        return fov

    def image_shift_pixels(self):

        # Projection center shift from middle of viewport for each eye.
        # Taken from Oculus SDK Overview, page 24
        # Center should be at lens separation / 2 instead of viewport width / 2.
        p = self.parameters
        w = 0.5*p['HScreenSize']                # meters
        s = p['LensSeparationDistance']         # meters
        dx = 0.5*s - 0.5*w                      # meters
        ppm = p['HResolution'] / p['HScreenSize']       # pixels per meter
        xp = dx * ppm
        return xp

    def radial_warp_parameters(self):

        p = self.parameters
        return p['DistortionK']

    def use_oculus_orientation(self):

        from . import _oculus
        q = _oculus.state()
        w,x,y,z = q     # quaternion orientation = (cos(a/2), axis * sin(a/2))

        from math import sqrt, atan2, pi
        vn = sqrt(x*x+y*y+z*z)
        a = 2*atan2(vn, w)
        axis = (x/vn, y/vn, z/vn) if vn > 0 else (0,0,1)
        from ...geometry import place
        r = place.rotation(axis, a*180/pi)
        if not self.last_rotation is None:
            rdelta = self.last_rotation.inverse()*r
            c = self.view.camera
            c.set_view(c.view()*rdelta)
        self.last_rotation = r

def start_oculus(session):
    start = False
    oht = session.oculus
    if oht is None:
        oht = Oculus_Head_Tracking()
        view = session.view
        success = oht.start_event_processing(view)
        session.show_status('started oculus head tracking ' + ('success' if success else 'failed'))
        if success:
            session.oculus = oht
            c = view.camera
            from math import pi
            c.field_of_view = oht.field_of_view() * 180 / pi
            c.eye_separation_scene = 0.2        # TODO: This is good value for inside a molecule, not for far from molecule.
            c.eye_separation_pixels = 2*oht.image_shift_pixels()
            view.set_camera_mode('oculus')
            view.render.radial_warp_coefficients = oht.radial_warp_parameters()
            print ('Radial warp', oht.radial_warp_parameters())
            start = True
        else:
            oht = None

    d = session.application.desktop()
    mw = session.main_window
    if start or d.screenNumber(mw) == d.primaryScreen():
        mw.toolbar.hide()
        mw.command_line.hide()
        mw.statusBar().hide()
        w,h = oht.display_size()
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

