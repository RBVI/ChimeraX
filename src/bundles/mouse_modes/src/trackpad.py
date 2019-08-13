# vim: set expandtab ts=4 sw=4:

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

class MultitouchTrackpad:
    '''
    Make two finger drag on Mac trackpad rotate scene,
    and three finger drag translate scene,
    and two finger pinch zoom scene.
    '''
    def __init__(self, session):
        self._session = session
        self._view = session.main_view
        self._recent_touches = []	# List of Touch instances
        self._last_touch_locations = {}	# Map touch id -> (x,y)
        from .settings import settings
        self.trackpad_speed = settings.trackpad_sensitivity   	# Trackpad position sensitivity
        # macOS trackpad units are in points (1/72 inch).
        cm_tpu = 72/2.54		# Convert centimeters to trackpad units.
        self._full_rotation_distance = 6 * cm_tpu		# trackpad units
        self._full_width_translation_distance = 6 * cm_tpu      # trackpad units
        self._zoom_scaling = 3		# zoom (z translation) faster than xy translation.
        self._twist_scaling = 6		# twist faster than finger rotation
        self._wheel_click_pixels = 5	# number of pixels drag that equals one scroll wheel click
        self._touch_handler = None
        self._received_touch_event = False

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
        from PyQt5.Qt import Qt
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

        from PyQt5.QtCore import QEvent
        t = event.type()
        if t == QEvent.TouchUpdate:
            # On Mac touch events get backlogged in queue when the events cause 
            # time consuming computatation.  It appears Qt does not collapse the events.
            # So event processing can get tens of seconds behind.  To reduce this problem
            # we only handle the most recent touch update per redraw.
            self._recent_touches = [Touch(t) for t in event.touchPoints()]
        elif t == QEvent.TouchEnd or t == QEvent.TouchCancel or t == QEvent.TouchBegin:
            # Sometimes we don't get a TouchEnd macOS gesture like 3-finger swipe up
            # for mission control took over half-way through a gesture.  So also
            # remove old touches when we get a begin.
            self._recent_touches = []
            self._last_touch_locations.clear()

    def _collapse_touch_events(self):
        touches = self._recent_touches
        if touches:
            self._process_touches(touches)
            self._recent_touches = []

    def _process_touches(self, touches):
        n = len(touches)
        speed = self.trackpad_speed
        moves = [t.move(self._last_touch_locations) for t in touches]
        if n == 2:
            (dx0,dy0),(dx1,dy1) = moves[0], moves[1]
            from math import sqrt, exp, atan2, pi
            l0,l1 = sqrt(dx0*dx0 + dy0*dy0),sqrt(dx1*dx1 + dy1*dy1)
            d12 = dx0*dx1+dy0*dy1
            if d12 < 0:
                # Finger moving in opposite directions: pinch or twist
                (x0,y0),(x1,y1) = [(t.x,t.y) for t in touches[:2]]
                sx,sy = x1-x0,y1-y0
                sn = sqrt(sx*sx + sy*sy)
                sd0,sd1 = sx*dx0 + sy*dy0, sx*dx1 + sy*dy1
                if abs(sd0) > 0.5*sn*l0 and abs(sd1) > 0.5*sn*l1:
                    # Fingers move along line between them: pinch to zoom
                    zf = 1 + speed * self._zoom_scaling * (l0+l1) / self._full_width_translation_distance
                    if sd1 < 0:
                        zf = 1/zf
                    self._zoom(zf)
                else:
                    # Fingers move perpendicular to line between them: twist
                    rot = atan2(-sy*dx1+sx*dy1,sn*sn) + atan2(sy*dx0-sx*dy0,sn*sn)
                    a = -speed * self._twist_scaling * rot * 180 / pi
                    zaxis = (0,0,1)
                    self._rotate(zaxis, a)
                return
            # Fingers moving in same direction: rotation
            dx = sum(x for x,y in moves)/n
            dy = sum(y for x,y in moves)/n
            from math import sqrt
            turns = sqrt(dx*dx + dy*dy)/self._full_rotation_distance
            angle = speed*360*turns
            self._rotate((dy, dx, 0), angle)
        elif n == 3:
            dx = sum(x for x,y in moves)/n
            dy = sum(y for x,y in moves)/n
            ww = self._view.window_size[0]	# Window width in pixels
            s = speed * ww / self._full_width_translation_distance
            self._translate((s*dx, -s*dy, 0))
        elif n == 4:
            # Use scrollwheel mouse mode
            ses = self._session
            from .mousemodes import keyboard_modifier_names, MouseEvent
            modifiers = keyboard_modifier_names(ses.ui.queryKeyboardModifiers())
            scrollwheel_mode = ses.ui.mouse_modes.mode(button = 'wheel', modifiers = modifiers)
            if scrollwheel_mode:
                xy = (sum(t.x for t in touches)/n, sum(t.y for t in touches)/n)
                dy = sum(y for x,y in moves)/n			# pixels
                delta = speed * dy / self._wheel_click_pixels	# wheel clicks
                scrollwheel_mode.wheel(MouseEvent(position = xy, wheel_value = delta, modifiers = modifiers))

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
        from a non-trackpad device.  Unfortunately macOS and Qt5 provides no way to distinguish
        magic mouse scroll from a trackpad scroll, both are reported as synthesized events
        and there is no source device id available.  If there are any trackpad devices and
        multitouch is enabled then the magic mouse wheel events are thrown away.  This
        is ChimeraX bug #1474.  We used to instead see if we recently received a touch event
        and in that case throw away any wheel event.  Unfortunately on macOS we sometimes
        don't get touch events after clicking in the Log or other windows and then moving
        back to the graphics window, until the second multitouch drag is done.
        '''
        if self._touch_handler is None:
            return False	# Multi-touch disabled

        from PyQt5.QtCore import Qt
        if event.source() == Qt.MouseEventNotSynthesized:
            return False	# Event is from a real mouse.

        from PyQt5.QtGui import QTouchDevice
        if len(QTouchDevice.devices()) == 0:
            return False	# No trackpad devices

        return self._received_touch_event

class Touch:
    def __init__(self, touch_point):
        t = touch_point
        self.id = t.id()
        # Touch positions in macOS correspond to physical trackpad distances in points (= 1/72 inch).
        # There is an offset in Qt 5.9 which is the current pointer window position x,y (in pixels).
        self.x = t.pos().x()
        self.y = t.pos().y()
        self.last_x = t.lastPos().x()
        self.last_y = t.lastPos().y()

    def move(self, last_touch_locations):
        id = self.id
        if id in last_touch_locations:
            lx,ly = last_touch_locations[id]
        else:
            lx,ly = self.last_x, self.last_y
        x,y = self.x, self.y
        last_touch_locations[id] = (x,y)
        return (x-lx, y-ly)
