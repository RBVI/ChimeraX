# vi: set expandtab ts=4 sw=4:

class MouseModes:

    def __init__(self, graphics_window, session):

        self.graphics_window = graphics_window
        self.session = session
        self.view = graphics_window.view

        self.mouse_modes = {}           # Maps 'left', 'middle', 'right', 'wheel', 'pause' to MouseMode instance

        # Mouse pause parameters
        self.last_mouse_time = None
        self.mouse_pause_interval = 0.5         # seconds
        self.mouse_pause_position = None

        self.bind_standard_mouse_modes()

        self.set_mouse_event_handlers()

    def set_mouse_event_handlers(self):
        import wx
        gw = self.graphics_window
        gw.opengl_canvas.Bind(wx.EVT_LEFT_DOWN,
              lambda e: self.dispatch_mouse_event(e, "left", "mouse_down"))
        gw.opengl_canvas.Bind(wx.EVT_MIDDLE_DOWN,
              lambda e: self.dispatch_mouse_event(e, "middle", "mouse_down"))
        gw.opengl_canvas.Bind(wx.EVT_RIGHT_DOWN,
              lambda e: self.dispatch_mouse_event(e, "right", "mouse_down"))
        gw.opengl_canvas.Bind(wx.EVT_MOTION,
              lambda e: self.dispatch_mouse_event(e, None, "mouse_drag"))
        gw.opengl_canvas.Bind(wx.EVT_LEFT_UP,
              lambda e: self.dispatch_mouse_event(e, "left", "mouse_up"))
        gw.opengl_canvas.Bind(wx.EVT_MIDDLE_UP,
              lambda e: self.dispatch_mouse_event(e, "middle", "mouse_up"))
        gw.opengl_canvas.Bind(wx.EVT_RIGHT_UP,
              lambda e: self.dispatch_mouse_event(e, "right", "mouse_up"))
        gw.opengl_canvas.Bind(wx.EVT_LEFT_DCLICK,
              lambda e: self.dispatch_mouse_event(e, "left", "mouse_double"))
        gw.opengl_canvas.Bind(wx.EVT_MIDDLE_DCLICK,
              lambda e: self.dispatch_mouse_event(e, "middle", "mouse_double"))
        gw.opengl_canvas.Bind(wx.EVT_RIGHT_DCLICK,
              lambda e: self.dispatch_mouse_event(e, "right", "mouse_double"))
        gw.opengl_canvas.Bind(wx.EVT_MOUSEWHEEL, self.wheel_event)

    def dispatch_mouse_event(self, event, button, action):
        if action in ('mouse_down', 'mouse_double'):
            # remember button for later drag events
            self.graphics_window.opengl_canvas.CaptureMouse()
        elif action == 'mouse_drag':
            if not event.Dragging():
                return
        elif action == 'mouse_up':
            self.graphics_window.opengl_canvas.ReleaseMouse()
        if button is None:
            if event.LeftIsDown():
                button = "left"
            elif event.MiddleIsDown():
                button = "middle"
            elif event.RightIsDown():
                button = "right"
            else:
                return
            if not self.graphics_window.opengl_canvas.HasCapture():
                # a Windows thing; can lose mouse capture w/o mouse up
                return
        m = self.mouse_modes.get(button)
        if m and hasattr(m, action):
            f = getattr(m, action)
            f(MouseEvent(event))

    def cursor_position(self):
        import wx
        return self.graphics_window.ScreenToClient(wx.GetMousePosition())

    def bind_standard_mouse_modes(self, buttons = ('left', 'middle', 'right', 'wheel', 'pause')):
        modes = (
            ('left', RotateMouseMode),
            ('middle', TranslateMouseMode),
            ('right', ZoomMouseMode),
            ('wheel', ZoomMouseMode),
            ('pause', ObjectIdMouseMode),
            )
        s = self.session
        for button, mode_class in modes:
            if button in buttons:
                self.bind_mouse_mode(button, mode_class(s))

    # Button is "left", "middle", "right", "wheel", or "pause"
    def bind_mouse_mode(self, button, mode):
        self.mouse_modes[button] = mode

    def wheel_event(self, event):
        f = self.mouse_modes.get('wheel')
        if f:
            f.wheel(MouseEvent(event))

    def mouse_pause_tracking(self):
        cp = self.cursor_position()
        w,h = self.view.window_size
        x,y = cp
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
        m = self.mouse_modes.get('pause')
        if m:
            m.pause(self.mouse_pause_position)

class MouseMode:

    def __init__(self, session):
        self.session = session
        self.view = session.main_view

        self.mouse_down_position = None
        self.last_mouse_position = None

    def mouse_down(self, event):
        pos = event.position()
        self.mouse_down_position = pos
        self.last_mouse_position = pos

    def mouse_up(self, event):
        self.mouse_down_position = None
        self.last_mouse_position = None

    def mouse_motion(self, event):
        lmp = self.last_mouse_position
        x, y = pos = event.position()
        if lmp is None:
            dx = dy = 0
        else:
            dx = x - lmp[0]
            dy = y - lmp[1]
            # dy > 0 is downward motion.
        self.last_mouse_position = pos
        return dx, dy

    def wheel(self):
        pass

    def pause(self, position):
        pass

    def pixel_size(self, min_scene_frac = 1e-5):
        v = self.view
        psize = v.pixel_size()
        b = v.drawing_bounds()
        if not b is None:
            w = b.width()
            psize = max(psize, w*min_scene_frac)
        return psize

class RotateMouseMode(MouseMode):

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.mouse_perimeter = False

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        w,h = self.view.window_size
        cx, cy = x-0.5*w, y-0.5*h
        fperim = 0.9
        self.mouse_perimeter = (abs(cx) > fperim*0.5*w or abs(cy) > fperim*0.5*h)

    def mouse_up(self, event):
        if event.position() == self.mouse_down_position:
            self.mouse_select(event)
        MouseMode.mouse_up(self, event)

    def mouse_drag(self, event):
        axis, angle = self.mouse_rotation(event)
        self.rotate(axis, angle)

    def rotate(self, axis, angle):
        v = self.view
        # Convert axis from camera to scene coordinates
        saxis = v.camera.position.apply_without_translation(axis)
        v.rotate(saxis, angle, self.models())

    def mouse_rotation(self, event):

        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5*math.sqrt(dx*dx+dy*dy)
        if self.mouse_perimeter:
            # z-rotation
            axis = (0,0,1)
            w, h = self.view.window_size
            x, y = event.position()
            ex, ey = x-0.5*w, y-0.5*h
            if -dy*ex+dx*ey < 0:
                angle = -angle
        else:
            axis = (dy,dx,0)
        return axis, angle

    def models(self):
        return None

    def mouse_select(self, event):

        x,y = event.position()
        v = self.view
        pick = v.first_intercept(x,y)
        ses = self.session
        toggle = event.shift_down()
        if pick is None:
            if not toggle:
                ses.selection.clear()
                ses.logger.status('cleared selection')
        else:
            if not toggle:
                ses.selection.clear()
            pick.select(toggle)
        ses.selection.clear_hierarchy()

class RotateSelectedMouseMode(RotateMouseMode):

    def models(self):
        return top_selected(self.session)

def top_selected(session):
    # Don't include parents of selected models.
    mlist = [m for m in session.selection.models()
             if (len(m.child_models()) == 0 or m.selected) and not any_parent_selected(m)]
    return None if len(mlist) == 0 else mlist

def any_parent_selected(m):
    if not hasattr(m, 'parent') or m.parent is None:
        return False
    return m.parent.selected or any_parent_selected(m.parent)

class TranslateMouseMode(MouseMode):

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        self.translate((dx, -dy, 0))

    def translate(self, shift):

        psize = self.pixel_size()
        s = tuple(dx*psize for dx in shift)     # Scene units
        v = self.view
        step = v.camera.position.apply_without_translation(s)    # Scene coord system
        v.translate(step, self.models())

    def models(self):
        return None

class TranslateSelectedMouseMode(TranslateMouseMode):

    def models(self):
        return top_selected(self.session)

class ZoomMouseMode(MouseMode):

    def mouse_drag(self, event):        

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        v = self.view
        shift = v.camera.position.apply_without_translation((0, 0, 3*psize*dy))
        v.translate(shift)

    def wheel(self, event):
        import sys
        d = event.wheel_value()
        psize = self.pixel_size()
        v = self.view
        shift = v.camera.position.apply_without_translation((0, 0, 100*d*psize))
        v.translate(shift)

class ObjectIdMouseMode(MouseMode):

    def pause(self, position):
        x,y = position
        p = self.view.first_intercept(x,y)
        if p:
            self.session.logger.status('Mouse over %s' % p.description())
        # TODO: Clear status if it is still showing mouse over message but mouse is over nothing.
        #      Don't want to clear a different status message, only mouse over message.

class MouseEvent:
    def __init__(self, event):
        self.event = event

    def shift_down(self):
        return self.event.ShiftDown()

    def position(self):
        return self.event.GetPosition()

    def wheel_value(self):
        return self.event.GetWheelRotation()/120.0   # Usually one wheel click is delta of 120
