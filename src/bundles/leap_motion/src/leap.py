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
# Command to use Leap Motion hand tracking with mouse and VR modes.
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
      Whether pinching emulates a mouse clicks (2D) or VR clicks (6D).
      In VR mode it emulates a VR button click including 3D position and
      rotation and the pointer (cross) moves and rotates in 3D.
      When emulating the mouse the left or right hand emulates a left button
      click or right button click at the position of the hand pointer and
      the pointer is shown in front of the models and moves in 2D.  Default "vr".
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
      LeapC 4.0 is optimized for head mounted, tracks considerably better than up facing.
    debug: bool
      Whether to send LeapC library debugging messages to console stderr.
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
        from numpy import array, float64
        self._center = array(center, float64)	# Center in leap motion device coordinates.
        self._facing = array(facing, float64)	# Sensor facing direction in screen coordinates.
        self._cord = array(cord, float64)	# Cord exit direction in screen coordinates.
        
    def scene_position_6d(self, hand_pose, view):
        from chimerax.geometry import translation, cross_product
        from chimerax.geometry import orthonormal_frame, inner_product
        
        # Adjust origin
        cpos = translation(-self._center) * hand_pose

        # Scale millimeters to scene units
        scene_center = view.center_of_rotation
        cam = view.camera
        factor = cam.view_width(scene_center) / self._width
        cpos = cpos.scale_translation(factor)		# millimeters to scene units

        # Rotate to screen orientation
        yaxis = self._facing
        zaxis = cross_product(yaxis, self._cord)
        cpos = orthonormal_frame(zaxis, ydir=yaxis) * cpos
        
        # Adjust depth origin to center of rotation.
        cam_pos = cam.position
        depth = inner_product(scene_center - cam_pos.origin(), -cam_pos.z_axis())
        cpos = translation((0,0,-depth)) * cpos	# Center in front of camera (-z axis).

        # Convert camera coordinates to scene coordinates
        scene_pos = cam_pos * cpos      # Map from camera to scene coordinates
        
        return scene_pos

    def scene_position_2d(self, hand_pose, view):
        # Adjust origin
        cpos = hand_pose.origin() - self._center

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
                 mode_left = 'vr', mode_right = 'vr',
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
        
        from .leap_cpp import leap_open
        self._connection = leap_open(head_mounted = head_mounted, debug = debug)
        self._new_frame_handler = session.triggers.add_handler('new frame', self._new_frame)

    def delete(self):
        self.session.triggers.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
        from .leap_cpp import leap_close
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
        from .leap_cpp import leap_hand_state
        for hnum, h in enumerate(self._hands):
            state = leap_hand_state(self._connection, hnum, self._max_delay)
            if state is None:
                h.display = False
            else:
                h.display = True
                palm_position, palm_normal, finger_direction, pinch_strength = state
                hand_pose = h._hand_pose(palm_position, palm_normal, finger_direction)
                h._position_pointer(hand_pose)
                h._update_pinch(pinch_strength, hand_pose)

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
        self._emulate = emulate			# 'mouse' or 'vr' for 2d or 6d mouse mode events
        self._hand = hand			# 'left' or 'right'
        self._tracking_space = tracking_space	# Maps device hand position to screen.
        self._pinch = False			# Are thumb and index finger touching?
        self._pinch_thresholds = pinch_thresholds	# Pinch strength on and off thresholds
        self._last_hand_pose = None		# Used for mouse mode drag events, leap device coordinates
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
        vax, nax = ry90 * cva, ry90.transform_vectors(cna)
        vay, nay = rx90 * cva, rx90.transform_vectors(cna)
        geom = [(vax,nax,cta), (vay,nay,cta)]
        if self._emulate == 'vr':
            geom.append((cva,cna,cta))  # z-axis
        va, na, ta = combine_geometry_vnt(geom)
        self.set_geometry(va, na, ta)

    def _hand_pose(self, palm_position, palm_normal, finger_direction):
        from chimerax.geometry import orthonormal_frame
        hp = orthonormal_frame(finger_direction, ydir = -palm_normal, origin = palm_position)
        return hp
        
    def _position_pointer(self, hand_pose):
        '''Hand position is in millimeters in Leap Motion coordinate system.'''
        self.position = self._scene_position(hand_pose)

    def _scene_position(self, hand_pose):
        '''
        Hand position is in millimeters in Leap Motion coordinate system.
        Scale hand range in millimeters to fill camera view.
        '''
        t = self._tracking_space
        view = self.session.main_view
        if self._emulate == 'mouse':
            scene_pos, win_xy = t.scene_position_2d(hand_pose, view)
            self._window_xy = win_xy
        elif self._emulate == 'vr':
            scene_pos = t.scene_position_6d(hand_pose, view)
        return scene_pos

    def _in_view(self, hand_pose):
        spos = self._scene_position(hand_pose).origin()
        return point_in_view(spos, self.session.main_view)
    
    def _update_pinch(self, pinch_strength, hand_pose):
        if self._pinch_changed(pinch_strength, hand_pose):
            self._send_pinch_event()
            self.color = self._pinch_color if self._pinch else self._unpinch_color
        if self._pinch:
            self._send_motion_event(hand_pose)
            self._last_hand_pose = hand_pose

    def _pinch_changed(self, pinch_strength, hand_pose):
        pon, poff = self._pinch_thresholds
        if pinch_strength >= pon:
            pinch = True
        elif pinch_strength <= poff:
            pinch = False
        else:
            pinch = self._pinch
        if pinch == self._pinch:
            return False
        if pinch and not self._in_view(hand_pose):
            # Suppress pinches out of view to reduce noise pinches
            # when hand is near edge of device field of view.
            return False
        self._pinch = pinch
        return True

    def _send_pinch_event(self):
        pinch = self._pinch
        if not pinch:
            self._last_hand_pose = None
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
            pos = self.position.origin()
            e = LeapPinchEvent(pos, self._far_clip_point(pos))
            if pinch and hasattr(mode, 'vr_press'):
                mode.vr_press(e)
            elif not pinch and hasattr(mode, 'vr_release'):
                mode.vr_release(e)

    def _far_clip_point(self, pos):
        view = self.session.main_view
        cam = view.camera
        view_num = 0
        cam_pos = cam.get_position(view_num).origin()
        cam_dir = cam.view_direction(view_num)
        pick_dir = pos - cam_pos
        near, far = view.near_far_distances(cam, view_num)
        from chimerax.geometry import inner_product
        denom = inner_product(pick_dir, cam_dir)
        d = (far - inner_product(cam_pos, cam_dir)) / denom if denom != 0 else 1000
        return cam_pos + d*pick_dir

    def _send_motion_event(self, hand_pose):
        mode = self._mouse_mode()
        if mode:
            if self._emulate == 'mouse':
                from chimerax.mouse_modes.mousemodes import MouseEvent
                event = MouseEvent(position = self._window_xy)
                mode.mouse_drag(event)
            elif self._emulate == 'vr' and hasattr(mode, 'vr_motion'):
                lpos = self._last_hand_pose
                if lpos is not None:
                    lspos = self._scene_position(lpos)
                    spos = self.position
                    move = spos * lspos.inverse()
                    vert = hand_pose.origin()[1] - lpos.origin()[1]
                    event = LeapMoveEvent(spos.origin(), move, vert)
                    mode.vr_motion(event)
            
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
    def __init__(self, position, pick_end_position):
        self._position = position
        self.tip_position = position
        self._pick_end_position = pick_end_position
    def picked_object(self, view):
        '''Return pick for object under position.'''
        xyz1, xyz2 = self.picking_segment()
        pick = view.picked_object_on_segment(xyz1, xyz2)
        return pick
    def picking_segment(self):
        return (self._position, self._pick_end_position)
    
class LeapMoveEvent:
    def __init__(self, position, move, room_vertical_motion):
        self.tip_position = position
        self.motion = move
        self.room_vertical_motion = (room_vertical_motion / 1000) * 2 # mm to m, plus 2x sensitivity
