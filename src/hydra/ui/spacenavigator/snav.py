# -----------------------------------------------------------------------------
#
class Space_Navigator:

    def __init__(self):

        self.speed = 1
        self.dominant = True    # Don't simultaneously rotate and translate
        self.fly_mode = False   # Control camera instead of models.
        self.view = None
        self.device = None
        self.processing_events = False

    def start_event_processing(self, view):

        if self.device is None:
            try:
                self.device = find_device()
            except:
                return False     # Connection failed.

        if self.device:
            self.view = view
            view.add_new_frame_callback(self.check_space_navigator)
            self.processing_events = True
            return True

        return False

    def stop_event_processing(self, view):

        if self.processing_events:
            view.remove_new_frame_callback(self.check_space_navigator)
            self.processing_events = False

    def check_space_navigator(self):

        e = self.device.last_event()
        if e is None:
            return
        rot, trans, buttons = e

        from math import sqrt

        # Rotation
        rx, ry, rz = rot         # 10 bits, signed, +/-512
        rmag = sqrt(rx*rx + ry*ry + rz*rz)

        # Translation
        tx, ty, tz = trans       # 10 bits, signed, +/-512
        tmag = sqrt(tx*tx + ty*ty + tz*tz)
        
        if self.dominant:
            if tmag < rmag:
                tmag = 0
            if rmag < tmag:
                rmag = 0
            if self.fly_mode:
                rt = 50
                if abs(ry) > abs(rx)+rt and abs(ry) > abs(rz)+rt: rx = rz = 0
                else: ry = 0
                rmag = sqrt(rx*rx + ry*ry + rz*rz)

        from numpy import array, float32
        from ...geometry import place

        if rmag > 0:
            axis = array((rx/rmag, ry/rmag, rz/rmag), float32)
            f = 3 if self.fly_mode else 30
            angle = self.speed*(f*rmag/512)
            rtf = place.rotation(axis, angle)
            self.apply_transform(rtf)

        if tmag > 0:
            axis = array((tx/tmag, ty/tmag, tz/tmag), float32)
            v = self.view
#            view_width = v.camera.view_width(v.center_of_rotation)
            b = v.bounds()
            if not b is None:
                f = .1 if self.fly_mode else 1
                view_width = b[1]
                shift = axis * 0.15 * self.speed * view_width * f * tmag/512
                ttf = place.translation(shift)
                self.apply_transform(ttf)

        if 'N1' in buttons or 31 in buttons:
            self.view_all()

        if 'N2' in buttons:
            self.toggle_dominant_mode()

    # Transform is in camera coordinates, with rotation about 0.
    def apply_transform(self, tf):

        v = self.view
        cv = v.camera.view()
        cvinv = v.camera.view_inverse()
        if self.fly_mode:
            cr = cvinv * v.camera.position()
            tf = tf.inverse()
        else:
            if tf.rotation_angle() > 1e-5:
                v.update_center_of_rotation()           # Rotation
            else:
                v.center_of_rotation_needs_update()     # Translation
            cr = cvinv * v.center_of_rotation
        from ...geometry.place import translation
        stf = cv * translation(cr) * tf * translation(-cr) * cvinv
        v.move(stf, update_clip_planes = True)

    def toggle_dominant_mode(self):

        self.dominant = not self.dominant
        from ..gui import show_status
        show_status('simultaneous rotation and translation: %s'
                    % (not self.dominant))

    def toggle_fly_mode(self):

        self.fly_mode = not self.fly_mode
        from chimera import viewer
        viewer.clipping = False
        from ..gui import show_status
        show_status('fly through mode: %s' % self.fly_mode)

    def view_all(self):

        self.view.view_all()

# -----------------------------------------------------------------------------
#
def find_device():

    from sys import platform
    if platform == 'darwin':
        from .snavmac import Space_Device_Mac
        return Space_Device_Mac()
    elif platform == 'win32':
        from .snavwin32 import Space_Device_Win32
        return Space_Device_Win32()
    elif platform[:5] == 'linux':
        from .snavlinux import Space_Device_Linux
        return Space_Device_Linux()

    return None

# -----------------------------------------------------------------------------
#
sn = None
def toggle_space_navigator(view):
    global sn
    if sn is None:
        sn = Space_Navigator()
        success = sn.start_event_processing(view)
        print('started space navigator', success)
    elif sn.processing_events:
        sn.stop_event_processing(view)
    else:
        sn.start_event_processing(view)

# -----------------------------------------------------------------------------
#
def toggle_fly_mode(view):
    global sn
    if sn is None:
        toggle_space_navigator(view)
    sn.fly_mode = not sn.fly_mode
