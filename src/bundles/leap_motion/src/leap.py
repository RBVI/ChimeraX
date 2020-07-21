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
def leapmotion(session, enable = None):
    '''
    Use Leap Motion hand tracking device for mouse modes.
    '''
    
    if enable is None:
        enable = True

    lm = getattr(session, '_leap_motion', None)
    if lm and lm.deleted:
        lm = None
    if enable:
        if lm is None:
            lm = LeapMotion(session)
            session._leap_motion = lm
            session.models.add([lm])
    elif not enable and lm:
        session.models.close([lm])
        delattr(session, '_leap_motion')
        
# -----------------------------------------------------------------------------
#
def register_leapmotion_command(logger):
    from chimerax.core.commands import register, create_alias, CmdDesc, BoolArg, IntArg, FloatArg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [],
                   synopsis = 'Enable leap motion hand tracker for mouse modes.',
                   url = 'help:user/commands/device.html#leapmotion')
    register('leapmotion', desc, leapmotion, logger=logger)
    create_alias('device leapmotion', 'leapmotion $*', logger=logger,
                 url='help:user/commands/device.html#leapmotion')

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class LeapMotion(Model):
    casts_shadows = False
    skip_bounds = True
    pickable = False
    SESSION_SAVE = False
    
    def __init__(self, session):
        Model.__init__(self, 'Leap Motion', session)

        self._max_delay = 0.2	# Don't use hand positions older than this (seconds).
        self._hands = [LeapHand(session, hand = 'left', emulate = 'mouse'),
                       LeapHand(session, hand = 'right', emulate = 'vr')]
        self.add(self._hands)	# Add hand models as children
        
        from ._leap import leap_open
        self._connection = leap_open()
        self._new_frame_handler = session.triggers.add_handler('new frame', self._new_frame)

    def delete(self):
        self.session.triggers.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
        from ._leap import leap_close
        leap_close(self._connection)
        self._connection = None
        Model.delete(self)
 
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
    
    def __init__(self, session, hand = 'right', emulate = 'mouse'):
        self._emulate = emulate			# 'mouse' or 'vr' for 2d or 3d mouse mode events
        self._hand = hand			# 'left' or 'right'
        self._hand_range = 250			# millimeters that span ChimeraX window.
        self._pinch = False			# Are thumb and index finger touching?
        self._pinch_thresholds = (0.9, 0.6)	# Pinch strength on and off thresholds
        self._last_position = None		# Used for mouse mode drag events, leap device coordinates
        self._window_xy = None			# Window position for mouse emulation

        name = self._hand + ' hand'
        Model.__init__(self, name, session)

        self._set_pointer_shape()

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

    def _update_pinch(self, pinch_strength, hand_position):
        if self._pinch_changed(pinch_strength):
            self._send_pinch_event()
            self.color = self._pinch_color if self._pinch else self._unpinch_color
        if self._pinch:
            self._send_motion_event(hand_position)
            self._last_position = hand_position

    def _position_pointer(self, hand_position):
        '''Hand position is in millimeters in Leap Motion coordinate system.'''
        self.position = self._scene_position(hand_position)

    def _scene_position(self, hand_position):
        '''
        Hand position is in millimeters in Leap Motion coordinate system.
        Scale hand range in millimeters to fill camera view.
        '''
        if self._emulate == 'mouse':
            scene_pos = self._scene_position_2d(hand_position)
        elif self._emulate == 'vr':
            scene_pos = self._scene_position_3d(hand_position)
        return scene_pos

    def _scene_position_2d(self, hand_position):
        view = self.session.main_view
        w, h = view.window_size
        r = self._hand_range
        win_xy = (w/2 + w*hand_position[0]/r, w*(1.5*r - hand_position[1])/r)
        self._window_xy = win_xy
        xyz_min, xyz_max = view.clip_plane_points(win_xy[0], win_xy[1])
        if xyz_min is not None and xyz_max is not None:
            scene_pos = .9*xyz_min + .1*xyz_max
        else:
            scene_pos = (0,0,0)
        rot = view.camera.position.zero_translation()
        from chimerax.geometry import translation
        return translation(scene_pos) * rot
        
    def _scene_position_3d(self, hand_position):
        # Adjust vertical origin
        y_center = self._hand_range	# vertical offset, millimeters
        cpos = hand_position - (0,y_center,0)

        # Scale millimeters to scene units
        view = self.session.main_view
        cofr = view.center_of_rotation
        cam = view.camera
        cam_pos = cam.position
        scale = cam.view_width(cofr) / self._hand_range
        cpos *= scale			# millimeters to scene units

        # Adjust depth origin to center of rotation.
        from chimerax.geometry import inner_product
        depth = inner_product(cofr - cam_pos.origin(), -cam_pos.z_axis())
        cpos[2] -= depth		# Center in front of camera (-z axis).

        # Convert camera coordinates to scene coordinates
        from chimerax.geometry import translation
        scene_pos = cam_pos * translation(cpos)      # Map from camera to scene coordinates
        
        return scene_pos

    def _pinch_changed(self, pinch_strength):
        pon, poff = self._pinch_thresholds
        if pinch_strength >= pon:
            pinch = True
        elif pinch_strength <= poff:
            pinch = False
        else:
            pinch = self._pinch
        if pinch == self._pinch:
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
