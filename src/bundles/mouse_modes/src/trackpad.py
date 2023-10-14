# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

class MultitouchTrackpad:
    '''
    Make two finger drag on Mac trackpad rotate scene,
    and three finger drag translate scene,
    and two finger pinch zoom scene.
    '''

    def __init__(self, session, mouse_mode_mgr):
        self._session = session
        self._mouse_mode_mgr = mouse_mode_mgr
        self._view = session.main_view
        self._recent_touches = []	# List of Touch instances
        self._modifier_keys = []
        self._last_touch_locations = {}	# Map touch id -> (x,y)
        from .settings import settings
        self.trackpad_speed = settings.trackpad_sensitivity   	# Trackpad position sensitivity
        # macOS trackpad units are in points (1/72 inch).
        cm_tpu = 72/2.54		# Convert centimeters to trackpad units.
        self._full_rotation_distance = 6 * cm_tpu		# trackpad units
        self._full_width_translation_distance = 6 * cm_tpu      # trackpad units
        self._zoom_scaling = 3		# zoom (z translation) faster than xy translation.
        self._twist_scaling = settings.trackpad_twist_speed	# twist faster than finger rotation
        self._wheel_click_pixels = 5	# number of pixels drag that equals one scroll wheel click
        self._touch_handler = None
        self._received_touch_event = False

    @property
    def full_width_translation_distance(self):
        return self._full_width_translation_distance

    @property
    def full_rotation_distance(self):
        return self._full_rotation_distance

    @property
    def wheel_click_pixels(self):
        return self._wheel_click_pixels

    def set_graphics_window(self, graphics_window):
        graphics_window.touchEvent = self._touch_event
        self._enable_touch_events(graphics_window)
        from .settings import settings
        self.enable_multitouch(settings.trackpad_multitouch)

    def enable_multitouch(self, enable):
        h = self._touch_handler
        t = self._session.triggers
        if enable:
            if h is None:
                h = t.add_handler('new frame', lambda tname, tdata: self._collapse_touch_events())
        elif h:
            t.remove_handler(h)
            h = None
        self._touch_handler = h

    def _enable_touch_events(self, graphics_window):
        from sys import platform
        if platform == 'darwin':
            # TODO: This hack works around QTBUG-53874 that touch events are not delivered on macOS.
            nsview_pointer = int(graphics_window.winId())
            from chimerax.core import _mac_util
            _mac_util.enable_multitouch(nsview_pointer)
        '''
        Various attempts to enable touch events from Python in Qt 5.9 all failed.
        print('graphics window winId', wid, int(wid))
        from Qt.QtCore import Qt
        w = self.widget
        wwid = w.winId()
        # wwid != wid so touch events are not enabled on graphics window.
        print('graphics widget winId', wwid, int(wwid))
        w.setAttribute(Qt.WA_AcceptTouchEvents)
        print('graphics widget touch enabled', w.testAttribute(Qt.WA_AcceptTouchEvents))
        '''

    # Appears that Qt has disabled touch events on Mac due to unresolved scrolling lag problems.
    # Searching for qt setAcceptsTouchEvents shows they were disabled Oct 17, 2012.
    # A patch that allows an environment variable QT_MAC_ENABLE_TOUCH_EVENTS to allow touch
    # events had status "Review in Progress" as of Jan 16, 2013 with no more recent update.
    # The Qt 5.0.2 source code qcocoawindow.mm does not include the environment variable patch.
    def _touch_event(self, event):

        if self._touch_handler is None:
            return

        self._received_touch_event = True

        from Qt.QtCore import QEvent
        t = event.type()
        # For some unfathomable reason the QTouchEvent.modifiers() method always
        # returns zero (QTBUG-60389, unresolved since 2017). So we need to do a
        # little hacky workaround

        from .mousemodes import decode_modifier_bits
        # session.ui.keyboardModifiers() does *not* work here (always returns 0)
        mb = self._session.ui.queryKeyboardModifiers()
        self._modifier_keys = decode_modifier_bits(mb)


        if t == QEvent.Type.TouchUpdate:
            # On Mac touch events get backlogged in queue when the events cause
            # time consuming computatation.  It appears Qt does not collapse the events.
            # So event processing can get tens of seconds behind.  To reduce this problem
            # we only handle the most recent touch update per redraw.
            self._recent_touches = [Touch(t) for t in event.points()]
        elif t == QEvent.Type.TouchEnd or t == QEvent.Type.TouchCancel or t == QEvent.Type.TouchBegin:
            # Sometimes we don't get a TouchEnd macOS gesture like 3-finger swipe up
            # for mission control took over half-way through a gesture.  So also
            # remove old touches when we get a begin.
            self._recent_touches = []
            self._last_touch_locations.clear()

    def _collapse_touch_events(self):
        touches = self._recent_touches
        if touches:
            event = self._process_touches(touches)
            self._recent_touches = []
            self._mouse_mode_mgr._dispatch_touch_event(event)

    def _process_touches(self, touches):
        min_pinch = 0.1
        pinch = twist = scroll = None
        two_swipe = None
        three_swipe = None
        four_swipe = None
        n = len(touches)
        speed = self.trackpad_speed
        position = (sum(t.x for t in touches)/n, sum(t.y for t in touches)/n)
        moves = [t.move(self._last_touch_locations) for t in touches]
        dx = sum(x for x,y in moves)/n
        dy = sum(y for x,y in moves)/n

        if n == 2:
            (dx0,dy0),(dx1,dy1) = moves[0], moves[1]
            from math import sqrt, exp, atan2, pi
            l0,l1 = sqrt(dx0*dx0 + dy0*dy0),sqrt(dx1*dx1 + dy1*dy1)
            d12 = dx0*dx1+dy0*dy1
            if l0 >= min_pinch and l1 >= min_pinch and d12 < -0.7*l0*l1:
                # Finger moving in opposite directions: pinch/twist
                (x0,y0),(x1,y1) = [(t.x,t.y) for t in touches[:2]]
                sx,sy = x1-x0,y1-y0
                sn = sqrt(sx*sx + sy*sy)
                sd0,sd1 = sx*dx0 + sy*dy0, sx*dx1 + sy*dy1
                if abs(sd0) > 0.5*sn*l0 and abs(sd1) > 0.5*sn*l1:
                    # pinch
                    zf = 1 + speed * self._zoom_scaling * (l0+l1) / self._full_width_translation_distance
                    if sd1 < 0:
                        zf = 1/zf
                    pinch = zf
                else:
                    # twist
                    rot = atan2(-sy*dx1+sx*dy1,sn*sn) + atan2(sy*dx0-sx*dy0,sn*sn)
                    a = -speed * self._twist_scaling * rot * 180 / pi
                    twist = a
            else:
                two_swipe = tuple([d/self._full_width_translation_distance for d in (dx, dy)])
                scroll = speed * dy / self._wheel_click_pixels
        elif n == 3:
            three_swipe = tuple([d/self._full_width_translation_distance for d in (dx, dy)])
        elif n == 4:
            four_swipe = tuple([d/self._full_width_translation_distance for d in (dx, dy)])

        return MultitouchEvent(modifiers=self._modifier_keys,
            position=position, wheel_value=scroll, two_finger_trans=two_swipe, two_finger_scale=pinch,
            two_finger_twist=twist, three_finger_trans=three_swipe,
            four_finger_trans=four_swipe)

        return pinch, twist, scroll, two_swipe, three_swipe, four_swipe

    def _rotate(self, screen_axis, angle):
        if angle == 0:
            return
        v = self._view
        axis = v.camera.position.transform_vector(screen_axis)	# Scene coords
        v.rotate(axis, angle)

    def _translate(self, screen_shift):
        v = self._view
        psize = v.pixel_size()
        s = tuple(dx*psize for dx in screen_shift)     # Scene units
        shift = v.camera.position.transform_vector(s)    # Scene coord system
        v.translate(shift)

    def _zoom(self, factor):
        v = self._view
        c = v.camera
        if c.name == 'orthographic':
            c.field_width = c.field_width / factor
            # TODO: Make camera field_width a property so it knows to redraw.
            c.redraw_needed = True
        else:
            psize = v.pixel_size()
            zpix = (factor-1) * v.window_size[0]	# Window width in pixels
            shift = v.camera.position.transform_vector((0,0,zpix*psize))    # Scene coord system
            v.translate(shift)

    def discard_trackpad_wheel_event(self, event):
        '''
        macOS generates mouse wheel events in response to a two-finger drag on trackpad.
        We discard those if we have multitouch trackpad support enabled so that 2-finger
        drag should be rotation.  But we don't want to discard wheel events that come
        from a non-trackpad device.  ChimeraX ticket #1474 and #9534'''
        if self._touch_handler is None:
            return False	# Multi-touch disabled

        d = event.pointingDevice()
        if d.type() == d.DeviceType.Mouse:
            return False	# Event is from a real mouse.

        if d.capabilities() & d.Capability.Scroll:
            return False	# Magic mouse and generic mouse has Scroll but Magic Trackpad does not
        
        from Qt.QtGui import QInputDevice
        touch_devices = [d for d in QInputDevice.devices() if d.type() == d.DeviceType.TouchPad]
        if len(touch_devices) == 0:
            return False	# No trackpad devices

        return self._received_touch_event

class Touch:
    def __init__(self, touch_point):
        t = touch_point
        self.id = t.id()
        # Touch positions in macOS correspond to physical trackpad distances in points (= 1/72 inch).
        # There is an offset in Qt 5.9 which is the current pointer window position x,y (in pixels).
        p = t.position()
        self.x = p.x()
        self.y = p.y()

    def move(self, last_touch_locations):
        id = self.id
        x,y = self.x, self.y
        if id in last_touch_locations:
            lx,ly = last_touch_locations[id]
            dx, dy = (x-lx, y-ly)
        else:
            dx = dy = 0
        last_touch_locations[id] = (x,y)
        return (dx, dy)

touch_action_to_property = {
    'pinch':    'two_finger_scale',
    'twist':    'two_finger_twist',
    'two finger swipe': 'two_finger_trans',
    'three finger swipe':   'three_finger_trans',
    'four finger swipe':    'four_finger_trans',
}


class MultitouchBinding:
    '''
    Associates an action on a multitouch trackpad and a set of modifier keys
    ('alt', 'command', 'control', 'shift') with a MouseMode.
    '''
    valid_actions = list(touch_action_to_property.keys())

    def __init__(self, action, modifiers, mode):
        if action not in self.valid_actions:
            from chimerax.core.errors import UserError
            raise UserError('Unrecognised touchpad action! Must be one of: {}'.format(
                ', '.join(self.valid_actions)
            ))
        self.action = action
        self.modifiers = modifiers
        self.mode = mode
    def matches(self, action, modifiers):
        '''
        Does this binding match the specified action and modifiers?
        A match requires all of the binding modifiers keys are among
        the specified modifiers (and possibly more).
        '''
        return (action==self.action and
            len([k for k in self.modifiers if not k in modifiers]) == 0
        )
    def exact_match(self, action, modifiers):
        '''
        Does this binding exactly match the specified action and modifiers?
        An exact match requires the binding modifiers keys are exactly the
        same set as the specified modifier keys.
        '''
        return action == self.action and set(modifiers) == set(self.modifiers)


from .mousemodes import MouseEvent
class MultitouchEvent(MouseEvent):
    '''
    Provides an interface to events fired by multi-touch trackpads and modifier
    keys so that mouse modes do not directly depend on details of the window
    toolkit or trackpad implementation.
    '''
    def __init__(self, modifiers = None, position=None, wheel_value = None,
            two_finger_trans=None, two_finger_scale=None, two_finger_twist=None,
            three_finger_trans=None, four_finger_trans=None):
        super().__init__(event=None, modifiers=modifiers, position=position, wheel_value=wheel_value)
        self._two_finger_trans = two_finger_trans
        self._two_finger_scale = two_finger_scale
        self._two_finger_twist = two_finger_twist
        self._three_finger_trans = three_finger_trans
        self._four_finger_trans = four_finger_trans

    @property
    def modifiers(self):
        return self._modifiers

    # @property
    # def event(self):
    #     '''
    #     The core QTouchEvent object
    #     '''
    #     return self._event

    @property
    def wheel_value(self):
        '''
        Supported API.
        Effective mouse wheel value if two-finger vertical swipe is to be
        interpreted as a scrolling action.
        '''
        return self._wheel_value

    @property
    def two_finger_trans(self):
        '''
        Supported API.
        Returns a tuple (delta_x, delta_y) in screen coordinates representing
        the movement when a two-finger swipe is interpreted as a translation
        action.
        '''
        return self._two_finger_trans

    @property
    def two_finger_scale(self):
        '''
        Supported API
        Returns a float representing the change in a two-finger pinching action.
        '''
        return self._two_finger_scale

    @property
    def two_finger_twist(self):
        '''
        Supported API
        Returns the rotation in degrees defined by a two-finger twisting action.
        '''
        return self._two_finger_twist

    @property
    def three_finger_trans(self):
        '''
        Supported API
        Returns a tuple (delta_x, delta_y) in screen coordinates representing
        the translation in a 3-fingered swipe.
        '''
        return self._three_finger_trans

    @property
    def four_finger_trans(self):
        '''
        Supported API
        Returns a tuple (delta_x, delta_y) in screen coordinates representing
        the translation in a 3-fingered swipe.
        '''
        return self._four_finger_trans
