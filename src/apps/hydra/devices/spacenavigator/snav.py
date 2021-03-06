# -----------------------------------------------------------------------------
#
class Space_Navigator:

    def __init__(self, view, log):

        self.view = view
        self.log = log

        self.speed = 1
        self.dominant = True    # Don't simultaneously rotate and translate
        self.fly_mode = False   # Control camera instead of models.
        self.device = None
        self.processing_events = False
        self.collision_map = None        # Volume data mask where camera cannot go
        self.button_1_action = 'view all'
        self.button_2_action = 'toggle dominant mode'

    def start_event_processing(self):

        if self.processing_events:
            return True

        if self.device is None:
            try:
                self.device = find_device()
            except:
                return False     # Connection failed.

        if self.device:
            self.view.add_new_frame_callback(self.check_space_navigator)
            self.processing_events = True
            return True

        return False

    def stop_event_processing(self):

        if self.processing_events:
            self.view.remove_new_frame_callback(self.check_space_navigator)
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
#            view_width = v.camera.view_width(v.center_of_rotation)
            b = self.view.drawing_bounds()
            if not b is None:
                f = .1 if self.fly_mode else 1
                view_width = b.xyz_max - b.xyz_min
                shift = axis * 0.15 * self.speed * view_width * f * tmag/512
                ttf = place.translation(shift)
                self.apply_transform(ttf)

        if 'N1' in buttons or 31 in buttons:
            self.button_pressed(self.button_1_action)

        if 'N2' in buttons:
            self.button_pressed(self.button_2_action)

    def button_pressed(self, action):

        if action == 'view all':
            self.view_all()
        elif action == 'toggle dominant mode':
            self.toggle_dominant_mode()
        elif action == 'toggle fly mode':
            self.toggle_fly_mode()

    # Transform is in camera coordinates, with rotation about 0.
    def apply_transform(self, tf):

        v = self.view
        cam = v.camera
        cp = cam.position
        cpinv = cp.inverse()
        if self.fly_mode:
            cr = cpinv * cam.position.origin()
            tf = tf.inverse()
        else:
            if tf.rotation_angle() > 1e-5:
                v.update_center_of_rotation()           # Rotation
            else:
                v.center_of_rotation_needs_update()     # Translation
            cr = cpinv * v.center_of_rotation
        from ...geometry.place import translation
        stf = cp * translation(cr) * tf * translation(-cr) * cpinv
        if self.collision(stf.inverse() * cam.position.origin()):
            return
        v.move(stf)

    def collision(self, xyz):
        cm = self.collision_map
        if cm is None:
            return False
        clev = max(cm.surface_levels)
        return (cm.interpolated_values([xyz], cm.position) >= clev)

    def toggle_dominant_mode(self):

        self.dominant = not self.dominant
        self.log.status('simultaneous rotation and translation: %s'
                        % (not self.dominant))

    def toggle_fly_mode(self):

        self.fly_mode = not self.fly_mode

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
def space_navigator(session):
    sn = session.space_navigator
    if sn is None:
        log = session
        sn = Space_Navigator(session.view, log)
        session.space_navigator = sn
    return sn

# -----------------------------------------------------------------------------
#
def toggle_space_navigator(session):
    sn = space_navigator(session)
    if sn.processing_events:
        sn.stop_event_processing()
    else:
        success = sn.start_event_processing()
        session.show_info('started space navigator: %s' % str(bool(success)))

# -----------------------------------------------------------------------------
#
def toggle_fly_mode(session):
    sn = space_navigator(session)
    sn.toggle_fly_mode()
    sn.button_2_action = 'toggle fly mode' if sn.fly_mode else 'toggle dominant mode'
    if not sn.processing_events:
        toggle_space_navigator(session)

# -----------------------------------------------------------------------------
#
def avoid_collisions(session):
    maps = session.maps()
    sn = session.space_navigator
    if sn is None and len(maps) > 0:
        toggle_space_navigator(session)
    if sn.collision_map is None and maps:
        sn.collision_map = maps[0]
    else:
        sn.collision_map = None

# -----------------------------------------------------------------------------
#
def snav_command(enable = None, fly = None, session = None):

    sn = space_navigator(session)
    if not enable is None:
        if enable:
            sn.start_event_processing()
        else:
            sn.stop_event_processing()
        
    if not fly is None:
        sn.fly_mode = bool(fly)
