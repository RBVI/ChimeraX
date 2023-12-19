# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
class Space_Navigator:

    def __init__(self, session):

        self.view = session.main_view
        self.log = session.logger
        self.ses_triggers = session.triggers

        self.speed = 1
        self.dominant = True    # Don't simultaneously rotate and translate
        self.fly_mode = False   # Control camera instead of models.
        self.device = None
        self.processing_events = False
        self.collision_map = None        # Volume data mask where camera cannot go
        self.button_1_action = 'view all'
        self.button_2_action = 'toggle dominant mode'
        self.error = None

    def start_event_processing(self):

        if self.processing_events:
            return True

        if self.device is None:
            try:
                self.device = find_device()
            except Exception as e:
                self.error = e
                return False     # Connection failed.

        if self.device:
            self.handler = self.ses_triggers.add_handler('new frame',
                self.check_space_navigator)
            self.processing_events = True
            return True

        from sys import platform
        self.error = 'ChimeraX does not support Space Navigator on %s' % platform
        return False

    def stop_event_processing(self):

        if self.processing_events:
            self.ses_triggers.remove_handler(self.handler)
            self.processing_events = False

    def check_space_navigator(self, *_):

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
        from chimerax.geometry import place

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

        if 'N1' in buttons or 31 in buttons or 'Left' in buttons:
            self.button_pressed(self.button_1_action)

        if 'N2' in buttons or 'Right' in buttons:
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
            if tf.rotation_angle() <= 1e-5:
                v._update_center_of_rotation = True  # Translation
            cr = cpinv * v.center_of_rotation
        from chimerax.geometry import translation
        stf = cp * translation(cr) * tf * translation(-cr) * cpinv
        if self.collision(stf.inverse() * cam.position.origin()):
            return
        v.move(stf)

    def collision(self, xyz):
        cm = self.collision_map
        if cm is None:
            return False
        clev = cm.maximum_surface_level
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
    if not hasattr(session, 'space_navigator') or session.space_navigator is None:
        session.space_navigator = Space_Navigator(session)
    return session.space_navigator

# -----------------------------------------------------------------------------
#
def toggle_space_navigator(session):
    sn = space_navigator(session)
    if sn.processing_events:
        sn.stop_event_processing()
    else:
        success = sn.start_event_processing()
        log = session.logger if hasattr(session, 'logger') else session # ChimeraX / Hydra compatibility
        log.info('started space navigator: %s' % str(bool(success)))

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
def device_snav(session, enable = None, fly = None, speed = None):
    '''Enable or disable moving models with Space Navigator input device.

    Parameters
    ----------
    enable : bool
      Enable (true) or disable (false) use of the Space Navigator device.
    fly : bool
      Enable flying mode where the Space Navigator motions control the camera,
      for example pushing forward flies the camera forward.  If fly is false,
      then the device controls the models, pushing forward would push the models
      away from the camera.  In both cases it is actually the camera that moves.
    speed : float
      Controls device sensitive, how fast the models move for a given device motion.
      Default 1.0.
    '''
    sn = space_navigator(session)
    if not enable is None:
        if enable:
            if not sn.start_event_processing():
                session.logger.warning('Could not start space navigator.\n\n%s' % sn.error)
        else:
            sn.stop_event_processing()
        
    if not fly is None:
        sn.fly_mode = bool(fly)

    if not speed is None:
        sn.speed = speed

# -----------------------------------------------------------------------------
# Register the snav command for ChimeraX.
#
def register_snav_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, register, FloatArg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('fly', BoolArg),
                              ('speed', FloatArg)],
                   synopsis = 'Turn on Space Navigator input device')
    register('device snav', desc, device_snav, logger=logger)
