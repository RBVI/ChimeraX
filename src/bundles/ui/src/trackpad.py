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
        self._recent_touch_points = None
        from chimerax.core.core_settings import settings
        self.trackpad_speed = settings.trackpad_sensitivity   	# Trackpad position sensitivity
        # macOS trackpad units are in points (1/72 inch).
        cm_tpu = 72/2.54		# Convert centimeters to trackpad units.
        self._full_rotation_distance = 4 * cm_tpu		# trackpad units
        self._full_width_translation_distance = 4 * cm_tpu      # trackpad units
        self._zoom_scaling = 3		# zoom (z translation) faster than xy translation.
        self._twist_scaling = 6		# twist faster than finger rotation
        self._last_trackpad_touch_time = 0
        self._last_trackpad_touch_count = 0
        self._touch_handler = None

    def set_graphics_window(self, graphics_window):
        graphics_window.touchEvent = self._touch_event
        self._enable_touch_events(graphics_window)
        from chimerax.core.core_settings import settings
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

        from PyQt5.QtCore import QEvent
        t = event.type()
        if t == QEvent.TouchUpdate:
            # On Mac touch events get backlogged in queue when the events cause 
            # time consuming computatation.  It appears Qt does not collapse the events.
            # So event processing can get tens of seconds behind.  To reduce this problem
            # we only handle the most recent touch update per redraw.
            self._recent_touch_points = event.touchPoints()
        elif t == QEvent.TouchEnd or t == QEvent.TouchCancel:
            self._last_trackpad_touch_count = 0
            self._recent_touch_points = None

    def _collapse_touch_events(self):
        touches = self._recent_touch_points
        if not touches is None:
            txy = [Touch(t) for t in touches]
            self._process_touches(txy)
            self._recent_touch_points = None

    def _process_touches(self, touches):
        n = len(touches)
        import time
        self._last_trackpad_touch_time = time.time()
        self._last_trackpad_touch_count = n
        speed = self.trackpad_speed
        moves = [t.move() for t in touches]
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
                    s = 1 if sd1 > 0 else -1
                    ww = self._view.window_size[0]	# Window width in pixels
                    zpix = s * speed * self._zoom_scaling * ww * (l0+l1) / self._full_width_translation_distance
                    self._translate((0,0,zpix))
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
            dy = sum(y for x,y in moves)/n
            ww = self._view.window_size[0]	# Window width in pixels
            zpix = speed * self._zoom_scaling * dy * ww / self._full_width_translation_distance
            self._translate((0, 0, zpix))

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

    def is_trackpad_wheel_event(self, event):
        # Suppress trackpad wheel events when using multitouch
        # Ignore scroll events generated by the Mac trackpad (2-finger drag).
        # There seems to be no reliable way to tell if a scroll came from the trackpad.
        # Scrolls from the Apple Magic Mouse look like a trackpad scroll.
        # Only way to tell true trackpad events seems to be to look at trackpad touches.
        if self._last_trackpad_touch_count >= 2:
            return True # Ignore trackpad generated scroll
        import time
        if time.time() < self._last_trackpad_touch_time + 1.0:
            # Suppress momentum scrolling for 1 second after trackpad scrolling ends.
            return True
        return False

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

    def move(self):
        return (self.x-self.last_x, self.y-self.last_y)
