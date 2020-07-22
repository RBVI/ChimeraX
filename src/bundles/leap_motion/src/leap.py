# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Command to view models in HTC Vive or Oculus Rift for ChimeraX.
#
def leapmotion(session, enable = None,
               pointer_size = None, mode_left = None, mode_right = None,
               width = None, center = None, facing = None, cord = None,
               pinch_thresholds = None, max_delay = None,
               head_mounted = False, debug = False):
    '''
    Use Leap Motion hand tracking device for mouse modes.
    The width, center, facing, and cord options specify sensitivity and
    how the Leap Motion device is positioned relative to the screen.

    Parameters
    ----------
    enable: bool
      Whether to turn the device on or off.  Default on.
    pointer_size: float
      Diameter of cross pointer indicating hand position.  Default 1.0.
    mode_left, mode_right: "mouse" or "vr"
      Whether pinching emulates a mouse clicks (2D) or VR clicks (3D).
      When emulating the mouse the left or right hand emulates a left button
      click or right button click at the position of the hand pointer.
      In VR mode it emulates a 3D button click.  In mouse mode the hand
      pointer is shown in front of the models and moves in 2D while in
      VR mode the hand pointer moves in 3D.
    width: float
      Hand motion range that corresponds to graphics window width.
      Millimeters.  Default 250.
    center: 3-tuple float
      Center of tracking space in Leap Device coordinates (y is sensor facing direction,
      x is device long axis away from cord).  Millimeters.  Default (0,250,0).
    facing: 3-tuple float
      Direction sensor faces relative to screen coordinates.
      Default (0,0,-1) if head mounted else (0,1,0).
    cord: 3-tuple float
      Direction sensor cord leaves device relative to screen coordinates.
      Default (1,0,0) if head mounted else (-1,0,0).
    pinch_thresholds: 2-tuple float
      Sets pinch detection sensitivity.  On and off pinch strengths on 0-1 scale.
      Default (0.9, 0.6).
    max_delay: float
      Don't use hand positions older than this many seconds.  Default 0.2.
    head_mounted: bool
      Whether device is head mounted (ie facing forward) or up facing.  Default True.
      CLeap 4.0 is optimized for head mounted, tracks considerably better than up facing.
    debug: bool
      Whether to send CLeap library debugging messages to console stderr.
    '''
    
    if enable is None:
        enable = True

    lm = getattr(session, '_leap_motion', None)
    if lm and lm.deleted:
        lm = None
    if enable:
        if lm is None:
            opts = {name:value for name, value in (('pointer_size',  pointer_size),
                                                   ('mode_left', mode_left),
                                                   ('mode_right', mode_right),
                                                   ('head_mounted', head_mounted),
                                                   ('debug', debug)) if value is not None}
            session._leap_motion = lm = LeapMotion(session, **opts)
            session.models.add([lm])
        if head_mounted:
            if facing is None:
                facing = (0,0,-1)
            if cord is None:
                cord = (1,0,0)
        lm.set_tracking_space(width, center, facing, cord)
        if pinch_thresholds is not None:
            lm.set_pinch_thresholds(pinch_thresholds)
        if max_delay is not None:
            lm.set_maximum_delay(max_delay)
    elif not enable and lm:
        session.models.close([lm])
        delattr(session, '_leap_motion')
        
# -----------------------------------------------------------------------------
#
def register_leapmotion_command(logger):
    from chimerax.core.commands import register, create_alias, CmdDesc
    from chimerax.core.commands import BoolArg, FloatArg, Float3Arg, Float2Arg, EnumOf
    ModeArg = EnumOf(('mouse', 'vr'))
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('pointer_size', FloatArg),
                              ('mode_left', ModeArg),
                              ('mode_right', ModeArg),
                              ('width', FloatArg),
                              ('center', Float3Arg),
                              ('facing', Float3Arg),
                              ('cord', Float3Arg),
                              ('pinch_thresholds', Float2Arg),
                              ('max_delay', FloatArg),
                              ('head_mounted', BoolArg),
                              ('debug', BoolArg)],
                   synopsis = 'Enable leap motion hand tracker for mouse modes.',
                   url = 'help:user/commands/device.html#leapmotion')
    register('leapmotion', desc, leapmotion, logger=logger)
    create_alias('device leapmotion', 'leapmotion $*', logger=logger,
                 url='help:user/commands/device.html#leapmotion')

# -----------------------------------------------------------------------------
#
class TrackingSpace:
    def __init__(self, width = 250, center = (0,250,0),
                 facing = (0,1,0), cord = (-1,0,0)):
        self._width = width	# Hand range corresponding to screen width.  Millimeters
        self._center = center	# Center in leap motion device coordinates.
        self._facing = facing	# Sensor facing direction in screen coordinates.
        self._cord = cord	# Cord exit direction in screen coordinates.
        
    def scene_position_3d(self, hand_position, view):
        # Adjust origin
        cpos = hand_position - self._center

        # Scale millimeters to scene units
        scene_center = view.center_of_rotation
        cam = view.camera
        scale = cam.view_width(scene_center) / self._width
        cpos *= scale			# millimeters to scene units

        # Rotate to screen orientation
        from chimerax.geometry import cross_product, orthonormal_frame
        yaxis = self._facing
        zaxis = cross_product(yaxis, self._cord)
        cpos = orthonormal_frame(zaxis, ydir=yaxis) * cpos
        
        # Adjust depth origin to center of rotation.
        cam_pos = cam.position
        from chimerax.geometry import inner_product
        depth = inner_product(scene_center - cam_pos.origin(), -cam_pos.z_axis())
        cpos[2] -= depth		# Center in front of camera (-z axis).

        # Convert camera coordinates to scene coordinates
        from chimerax.geometry import translation
        scene_pos = cam_pos * translation(cpos)      # Map from camera to scene coordinates
        
        return scene_pos

    def scene_position_2d(self, hand_position, view):
        # Adjust origin
        cpos = hand_position - self._center

        # Scale millimeters to scene units
        cpos /= self._width			# millimeters to unit width

        # Rotate to screen orientation
        from chimerax.geometry import cross_product, orthonormal_frame
        yaxis = self._facing
        zaxis = cross_product(yaxis, self._cord)
        cpos = orthonormal_frame(zaxis, ydir=yaxis) * cpos

        # Convert to window pixels
        w, h = view.window_size
        win_xy = (w/2 + w*cpos[0], h/2 - h*cpos[1])

        # Map to scene near front clip plane.
        xyz_min, xyz_max = view.clip_plane_points(win_xy[0], win_xy[1])
        scene_point = (0,0,0) if xyz_min is None else (.9*xyz_min + .1*xyz_max)

        # Convert camera coordinates to scene coordinates
        rot = view.camera.position.zero_translation()
        from chimerax.geometry import translation
        scene_pos = translation(scene_point) * rot

        return scene_pos, win_xy
        
# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class LeapMotion(Model):
    casts_shadows = False
    skip_bounds = True
    pickable = False
    SESSION_SAVE = False
    
    def __init__(self, session, pointer_size = 1.0,
                 mode_left = 'mouse', mode_right = 'vr',
                 pinch_thresholds = (0.9,0.6), max_delay = 0.2,
                 tracking_space = TrackingSpace(), head_mounted = False,
                 debug = False):
        Model.__init__(self, 'Leap Motion', session)

        self._tracking_space = tracking_space
        self._max_delay = max_delay	# Don't use hand positions older than this (seconds).
        self._hands = [LeapHand(session, hand = hand, emulate = mode,
                                pointer_size = pointer_size,
                                tracking_space = tracking_space,
                                pinch_thresholds = pinch_thresholds)
                       for hand, mode in (('left', mode_left), ('right', mode_right))]
        self.add(self._hands)	# Add hand models as children
        
        from ._leap import leap_open
        self._connection = leap_open(head_mounted = head_mounted, debug = debug)
        self._new_frame_handler = session.triggers.add_handler('new frame', self._new_frame)

    def delete(self):
        self.session.triggers.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
        from ._leap import leap_close
        leap_close(self._connection)
        self._connection = None
        Model.delete(self)

    def set_tracking_space(self, width = None, center = None, facing = None, cord = None):
        t = self._tracking_space
        for name, value in (('_width',width), ('_center',center),
                            ('_facing',facing), ('_cord',cord)):
            if value is not None:
                setattr(t, name, value)

    def set_pinch_thresholds(self, pinch_thresholds):
        for h in self._hands:
            h._pinch_thresholds = pinch_thresholds

    def set_maximum_delay(self, max_delay):
        self._max_delay = max_delay
 
    def _new_frame(self, tname, tdata):
        from ._leap import leap_hand_state
        for hnum, h in enumerate(self._hands):
            state = leap_hand_state(self._connection, hnum, self._max_delay)
            if state is None:
                h.display = False
            else:
                h.display = True
                hand_position, pinch_strength = state
                h._position_pointer(hand_position)
                h._update_pinch(pinch_strength, hand_position)

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class LeapHand(Model):
    casts_shadows = False
    skip_bounds = True
    pickable = False
    SESSION_SAVE = False
    
    def __init__(self, session, pointer_size = 1.0, hand = 'right', emulate = 'mouse',
                 tracking_space = TrackingSpace(), pinch_thresholds = (0.9, 0.6)):
        self._emulate = emulate			# 'mouse' or 'vr' for 2d or 3d mouse mode events
        self._hand = hand			# 'left' or 'right'
        self._tracking_space = tracking_space	# Maps device hand position to screen.
        self._pinch = False			# Are thumb and index finger touching?
        self._pinch_thresholds = pinch_thresholds	# Pinch strength on and off thresholds
        self._last_position = None		# Used for mouse mode drag events, leap device coordinates
        self._window_xy = None
        
        name = self._hand + ' hand'
        Model.__init__(self, name, session)

        self._set_pointer_shape(cross_diameter = pointer_size,
                                stick_diameter = 0.1*pointer_size)

        if hand == 'right':
            self._unpinch_color = (0,255,0,255)
            self._pinch_color = (0,255,255,255)
        else:
            self._unpinch_color = (255,255,0,255)
            self._pinch_color = (255,0,255,255)
        self.color = self._unpinch_color

    def _set_pointer_shape(self, cross_diameter = 1.0, stick_diameter = 0.1):
        from chimerax.surface import cylinder_geometry, combine_geometry_vnt
        cva, cna, cta = cylinder_geometry(radius = 0.5*stick_diameter, height = cross_diameter)
        from chimerax.geometry import rotation
        rx90, ry90 = rotation((1,0,0),90), rotation((0,1,0),90)
        vax, nax =  ry90 * cva, ry90.transform_vectors(cna)
        vay, nay =  rx90 * cva, rx90.transform_vectors(cna)
        va, na, ta = combine_geometry_vnt([(vax,nax,cta), (vay,nay,cta)])
        self.set_geometry(va, na, ta)

    def _position_pointer(self, hand_position):
        '''Hand position is in millimeters in Leap Motion coordinate system.'''
        self.position = self._scene_position(hand_position)

    def _scene_position(self, hand_position):
        '''
        Hand position is in millimeters in Leap Motion coordinate system.
        Scale hand range in millimeters to fill camera view.
        '''
        t = self._tracking_space
        view = self.session.main_view
        if self._emulate == 'mouse':
            scene_pos, win_xy = t.scene_position_2d(hand_position, view)
            self._window_xy = win_xy
        elif self._emulate == 'vr':
            scene_pos = t.scene_position_3d(hand_position, view)
        return scene_pos

    def _in_view(self, hand_position):
        spos = self._scene_position(hand_position).origin()
        return point_in_view(spos, self.session.main_view)
    
    def _update_pinch(self, pinch_strength, hand_position):
        if self._pinch_changed(pinch_strength, hand_position):
            self._send_pinch_event()
            self.color = self._pinch_color if self._pinch else self._unpinch_color
        if self._pinch:
            self._send_motion_event(hand_position)
            self._last_position = hand_position

    def _pinch_changed(self, pinch_strength, hand_position):
        pon, poff = self._pinch_thresholds
        if pinch_strength >= pon:
            pinch = True
        elif pinch_strength <= poff:
            pinch = False
        else:
            pinch = self._pinch
        if pinch == self._pinch:
            return False
        if pinch and not self._in_view(hand_position):
            # Suppress pinches out of view to reduce noise pinches
            # when hand is near edge of device field of view.
            return False
        self._pinch = pinch
        return True

    def _send_pinch_event(self):
        pinch = self._pinch
        if not pinch:
            self._last_position = None
        mode = self._mouse_mode()
        if mode is None:
            return
        if self._emulate == 'mouse':
            from chimerax.mouse_modes.mousemodes import MouseEvent
            event = MouseEvent(position = self._window_xy)
            if pinch:
                mode.mouse_down(event)
            else:
                mode.mouse_up(event)
        elif self._emulate == 'vr':
            if pinch and hasattr(mode, 'vr_press'):
                mode.vr_press(LeapPinchEvent(self.position.origin()))
            elif not pinch and hasattr(mode, 'vr_release'):
                mode.vr_release(LeapPinchEvent(self.position.origin()))

    def _send_motion_event(self, pos):
        mode = self._mouse_mode()
        if mode:
            if self._emulate == 'mouse':
                from chimerax.mouse_modes.mousemodes import MouseEvent
                event = MouseEvent(position = self._window_xy)
                mode.mouse_drag(event)
            elif self._emulate == 'vr' and hasattr(mode, 'vr_motion'):
                lpos = self._last_position
                if lpos is not None:
                    lspos = self._scene_position(lpos).origin()
                    from chimerax.geometry import translation
                    spos = self.position.origin()
                    shift = spos - lspos
                    move = translation(shift)
                    vert = pos[1] - lpos[1]
                    mode.vr_motion(LeapMoveEvent(spos, move, vert))
            
    def _mouse_mode(self):
        button = self._hand # left button for left hand, right button for right hand.
        return self.session.ui.mouse_modes.mode(button)

def point_in_view(point, view):
    w, h = view.window_size
    cam = view.camera
    # Compute planes bounding view
    planes = cam.rectangle_bounding_planes((0,0), (w,h), (w,h))
    from chimerax.geometry import inner_product
    for p in planes:
        if inner_product(point, p[:3]) + p[3] < 0:
            return False

    # Check near/far planes
    near, far = view.near_far_distances(cam, view_num=0)
    dist = inner_product(point - cam.position.origin(), cam.view_direction())
    if dist < near or dist > far:
        return False

    return True
    
class LeapPinchEvent:
    def __init__(self, position):
        self._position = position
        self.tip_position = position
    def picked_object(self, view):
        '''Return pick for object under position.'''
        # TODO: Compute clipped ray.
        xyz1 = self._position
        xyz2 = xyz1 + 100 * (xyz1 - view.camera.position.origin())
        from chimerax.mouse_modes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, view)
        return pick

class LeapMoveEvent:
    def __init__(self, position, move, room_vertical_motion):
        self.tip_position = position
        self.motion = move
        self.room_vertical_motion = (room_vertical_motion / 1000) * 2 # mm to m, plus 2x sensitivity
