class Mouse_Modes:

    def __init__(self, view):

        self.view = view
        self.mouse_modes = {}
        self.last_mouse_position = None
        self.last_mouse_time = None
        self.mouse_pause_interval = 0.5         # seconds
        self.mouse_pause_position = None
        self.mouse_perimeter = False
        self.wheel_function = None
        self.bind_standard_mouse_modes()

        view.mousePressEvent = self.mouse_press_event
        view.mouseMoveEvent = self.mouse_move_event
        view.mouseReleaseEvent = self.mouse_release_event
        view.wheelEvent = self.wheel_event

    # Button is "left", "middle", or "right"
    def bind_mouse_mode(self, button, mouse_down,
                        mouse_drag = None, mouse_up = None):
        self.mouse_modes[button] = (mouse_down, mouse_drag, mouse_up)
        
    def mouse_press_event(self, event):
        self.dispatch_mouse_event(event, 0)
    def mouse_move_event(self, event):
        self.dispatch_mouse_event(event, 1)
    def mouse_release_event(self, event):
        self.dispatch_mouse_event(event, 2)
    def wheel_event(self, event):
        f = self.wheel_function
        if f:
            f(event)
        
    def dispatch_mouse_event(self, event, fnum):

        b = self.event_button_name(event)
        f = self.mouse_modes.get(b)
        if f and f[fnum]:
            f[fnum](event)

    def event_button_name(self, event):

        # button() gives press/release button, buttons() gives move buttons
        b = event.button() | event.buttons()
        from .qt import QtCore
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

    def bind_standard_mouse_modes(self, buttons = ['left', 'middle', 'right', 'wheel']):
        modes = (
            ('left', self.mouse_down, self.mouse_rotate, self.mouse_up),
            ('middle', self.mouse_down, self.mouse_translate, self.mouse_up),
            ('right', self.mouse_down, self.mouse_contour_level, self.mouse_up),
            )
        for m in modes:
            if m[0] in buttons:
                self.bind_mouse_mode(*m)
        if 'wheel' in buttons:
            self.wheel_function = self.wheel_zoom

    def mouse_down(self, event):
        w,h = self.view.window_size
        cx, cy = event.x()-0.5*w, event.y()-0.5*h
        fperim = 0.9
        self.mouse_perimeter = (abs(cx) > fperim*0.5*w or abs(cy) > fperim*0.5*h)
        self.remember_mouse_position(event)

    def mouse_up(self, event):
        self.last_mouse_position = None

    def remember_mouse_position(self, event):
        from .qt import QtCore
        self.last_mouse_position = QtCore.QPoint(event.pos())

    def mouse_pause_tracking(self):
        v = self.view
        from .qt import QtGui
        cp = v.mapFromGlobal(QtGui.QCursor.pos())
        w,h = v.window_size
        x,y = cp.x(), cp.y()
        if x < 0 or y < 0 or x >= w or y >= h:
            return      # Cursor outside of graphics window
        from time import time
        t = time()
        mp = self.mouse_pause_position
        if cp == mp:
            lt = self.last_mouse_time
            if lt and t >= lt + self.mouse_pause_interval:
                self.mouse_pause()
                self.mouse_pause_position = None
                self.last_mouse_time = None
            return
        self.mouse_pause_position = cp
        if mp:
            # Require mouse move before setting timer to avoid
            # repeated mouse pause callbacks at same point.
            self.last_mouse_time = t

    def mouse_pause(self):
        lp = self.mouse_pause_position
        v = self.view
        p, s = v.first_intercept(lp.x(), lp.y())
        v.session.show_status('Mouse over %s' % s.description() if s else '')

    def mouse_motion(self, event):
        lmp = self.last_mouse_position
        if lmp is None:
            dx = dy = 0
        else:
            dx = event.x() - lmp.x()
            dy = event.y() - lmp.y()
            # dy > 0 is downward motion.
        self.remember_mouse_position(event)
        return dx, dy

    def mouse_rotate(self, event):

        axis, angle = self.mouse_rotation(event)
        v = self.view
        # Convert axis from camera to scene coordinates
        saxis = v.camera.view().apply_without_translation(axis)
        v.rotate(saxis, angle)

    def mouse_rotation(self, event):

        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5*math.sqrt(dx*dx+dy*dy)
        if self.mouse_perimeter:
            # z-rotation
            axis = (0,0,1)
            w, h = self.view.window_size
            ex, ey = event.x()-0.5*w, event.y()-0.5*h
            if -dy*ex+dx*ey < 0:
                angle = -angle
        else:
            axis = (dy,dx,0)
        return axis, angle

    def mouse_translate(self, event):

        dx, dy = self.mouse_motion(event)
        v = self.view
        psize = v.pixel_size()
        shift = v.camera.view().apply_without_translation((psize*dx, -psize*dy, 0))
        v.translate(shift)

    def mouse_translate_selected(self, event):

        v = self.view
        models = v.session.selected
        if models:
            dx, dy = self.mouse_motion(event)
            psize = v.pixel_size()
            shift = v.camera.view().apply_without_translation((psize*dx, -psize*dy, 0))
            v.translate(shift, models)

    def mouse_rotate_selected(self, event):

        v = self.view
        models = v.session.selected
        if models:
            axis, angle = self.mouse_rotation(event)
            # Convert axis from camera to scene coordinates
            saxis = v.camera.view().apply_without_translation(axis)
            v.rotate(axis, angle, models)

    def mouse_zoom(self, event):        

        dx, dy = self.mouse_motion(event)
        v = self.view
        psize = v.pixel_size()
        shift = v.camera.view().apply_without_translation((0, 0, 3*psize*dy))
        v.translate(shift)

    def wheel_zoom(self, event):        

        d = event.angleDelta().y()/120.0   # Usually one wheel click is delta of 120
        v = self.view
        psize = v.pixel_size()
        shift = v.camera.view().apply_without_translation((0, 0, 100*d*psize))
        v.translate(shift)
        
    def mouse_contour_level(self, event):

        dx, dy = self.mouse_motion(event)
        f = -0.001*dy
        
        models = self.view.session.model_list()
        from ..map.volume import Volume
        for m in models:
            if isinstance(m, Volume):
                adjust_threshold_level(m, f)
                m.show()
        
    def wheel_contour_level(self, event):
        d = event.angleDelta().y()       # Usually one wheel click is delta of 120
        f = d/(120.0 * 30)
        models = self.view.session.model_list()
        for m in models:
            adjust_threshold_level(m, f)
            m.show()

    # Appears that Qt has disabled touch events on Mac due to unresolved scrolling lag problems.
    # Searching for qt setAcceptsTouchEvents shows they were disabled Oct 17, 2012.
    # A patch that allows an environment variable QT_MAC_ENABLE_TOUCH_EVENTS to allow touch
    # events had status "Review in Progress" as of Jan 16, 2013 with no more recent update.
    # The Qt 5.0.2 source code qcocoawindow.mm does not include the environment variable patch.
    def trackpad_event(self, event):

        from .qt import QtCore, QtOpenGL
        t = event.type()
        print ('event', int(t))
        if t == QtCore.QEvent.TouchBegin:
            print ('touch begin')
        elif t == QtCore.QEvent.TouchUpdate:
            print ('touch update')
        elif t == QtCore.QEvent.TouchEnd:
            print ('touch end')
        return QtOpenGL.QGLWidget.event(self, event)

def adjust_threshold_level(m, f):
    ms = m.matrix_value_statistics()
    step = f * (ms.maximum - ms.minimum)
    if m.representation == 'solid':
        new_levels = [(l+step,b) for l,b in m.solid_levels]
        l,b = new_levels[-1]
        new_levels[-1] = (max(l,1.01*ms.maximum),b)
        m.set_parameters(solid_levels = new_levels)
    else:
        new_levels = tuple(l+step for l in m.surface_levels)
        m.set_parameters(surface_levels = new_levels)
