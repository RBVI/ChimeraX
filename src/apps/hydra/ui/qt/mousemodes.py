from .qt import QtCore, QtGui
from ..mousemodes import MouseModes

class QtMouseModes(MouseModes):

    def __init__(self, graphics_window):
        MouseModes.__init__(self, graphics_window)
        self.view.add_new_frame_callback(self.collapse_touch_events)
        self.recent_touch_points = None

    def set_mouse_event_handlers(self):
        gw = self.graphics_window
        gw.mousePressEvent = self.mouse_press_event
        gw.mouseMoveEvent = self.mouse_move_event
        gw.mouseReleaseEvent = self.mouse_release_event
        gw.wheelEvent = self.wheel_event
        gw.touchEvent = self.touch_event
        
    def mouse_press_event(self, event):
        self.dispatch_mouse_event(event, 0)
    def mouse_move_event(self, event):
        self.dispatch_mouse_event(event, 1)
    def mouse_release_event(self, event):
        self.dispatch_mouse_event(event, 2)
        
    def dispatch_mouse_event(self, event, fnum):

        b = self.event_button_name(event)
        f = self.mouse_modes.get(b)
        if f and f[fnum]:
            f[fnum](event)

    def event_button_name(self, event):

        # button() gives press/release button, buttons() gives move buttons
        b = event.button() | event.buttons()
        if b & QtCore.Qt.LeftButton:
            m = event.modifiers()
            if m == QtCore.Qt.AltModifier:
                bname = 'middle'
            elif m == QtCore.Qt.ControlModifier:
                # On Mac the Command key produces the Control modifier
                # and it is documented in Qt to behave that way.  Yuck.
                bname = 'right'
            else:
                bname = 'left'
        elif b & QtCore.Qt.MiddleButton:
            bname = 'middle'
        elif b & QtCore.Qt.RightButton:
            bname = 'right'
        else:
            bname = None
        return bname

    def shift_down(self, event):
        return bool(event.modifiers() & QtCore.Qt.ShiftModifier)

    def event_position(self, event):
        return event.x(), event.y()

    def cursor_position(self):
        p = self.graphics_window.mapFromGlobal(QtGui.QCursor.pos())
        return p.x(), p.y()

    def wheel_value(self, event):
        return event.angleDelta().y()/120.0   # Usually one wheel click is delta of 120

    # Appears that Qt has disabled touch events on Mac due to unresolved scrolling lag problems.
    # Searching for qt setAcceptsTouchEvents shows they were disabled Oct 17, 2012.
    # A patch that allows an environment variable QT_MAC_ENABLE_TOUCH_EVENTS to allow touch
    # events had status "Review in Progress" as of Jan 16, 2013 with no more recent update.
    # The Qt 5.0.2 source code qcocoawindow.mm does not include the environment variable patch.
    def touch_event(self, event):

        t = event.type()
        if t == QtCore.QEvent.TouchUpdate:
            # On Mac touch events get backlogged in queue when the events cause 
            # time consuming computatation.  It appears Qt does not collapse the events.
            # So event processing can get tens of seconds behind.  To reduce this problem
            # we only handle one touch update per redraw.
            self.recent_touch_points = event.touchPoints()
#            self.process_touches(event.touchPoints())
        elif t == QtCore.QEvent.TouchEnd:
            self.last_trackpad_touch_count = 0
            self.recent_touch_points = None
            self.mouse_up(event = None)

    def collapse_touch_events(self):
        touches = self.recent_touch_points
        if not touches is None:
            txy = [(t.id(), t.pos().x(), t.pos().y(), t.lastPos().x(), t.lastPos().y()) for t in touches]
            self.process_touches(txy)
            self.recent_touch_points = None

    def trackpad_event(self, dx, dy):
        p = self.last_mouse_position
        if p is None:
            x,y = cp = self.cursor_position()
        else:
            x,y = p[0]+dx, p[1]+dy
        class Trackpad_Event:
            def __init__(self,x,y):
                self._x, self._y = x,y
            def x(self):
                return self._x
            def y(self):
                return self._y
            def pos(self):
                return (self._x,self._y)
        e = Trackpad_Event(x,y)
        return e
