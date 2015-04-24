from .qt import QtCore, QtGui
from ..mousemodes import MouseModes

class QtMouseModes(MouseModes):

    def __init__(self, graphics_window):
        MouseModes.__init__(self, graphics_window, graphics_window.session)
        self.view.add_new_frame_callback(self.collapse_touch_events)
        self.recent_touch_points = None
        self.MouseEvent = MouseEvent

    def set_mouse_event_handlers(self):
        gw = self.graphics_window
        gw.mousePressEvent = self.mouse_press_event
        gw.mouseMoveEvent = self.mouse_move_event
        gw.mouseReleaseEvent = self.mouse_release_event
        gw.wheelEvent = self.wheel_event
        gw.touchEvent = self.touch_event
        
    def mouse_press_event(self, event):
        self.dispatch_mouse_event(event, 'mouse_down')
    def mouse_move_event(self, event):
        self.dispatch_mouse_event(event, 'mouse_drag')
    def mouse_release_event(self, event):
        self.dispatch_mouse_event(event, 'mouse_up')
        
    def dispatch_mouse_event(self, event, action):

        b = self.event_button_name(event)
        m = self.mouse_modes.get(b)
        if m:
            f = getattr(m, action)
            f(MouseEvent(event))

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

    def cursor_position(self):
        p = self.graphics_window.mapFromGlobal(QtGui.QCursor.pos())
        return p.x(), p.y()

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
            from ..mousemodes import MouseMode
            for m in self.mouse_modes.values():
                MouseMode.mouse_up(m, event = None)

    def collapse_touch_events(self):
        touches = self.recent_touch_points
        if not touches is None:
            txy = [(t.id(), t.pos().x(), t.pos().y(), t.lastPos().x(), t.lastPos().y()) for t in touches]
            self.process_touches(txy)
            self.recent_touch_points = None

    def trackpad_event(self, dx, dy, p):
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
            def position(self):
                return (self._x,self._y)
            def shift_down(self):
                return False
        e = Trackpad_Event(x,y)
        return e

class MouseEvent:
    def __init__(self, event):
        self.event = event

    def shift_down(self):
        return bool(self.event.modifiers() & QtCore.Qt.ShiftModifier)

    def position(self):
        return self.event.x(), self.event.y()

    def wheel_value(self):
        return self.event.angleDelta().y()/120.0   # Usually one wheel click is delta of 120
