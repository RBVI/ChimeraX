# vim: set expandtab ts=4 sw=4:

class MouseModes:

    def __init__(self, graphics_window, session):

        self.graphics_window = graphics_window
        self.session = session

        from .. import map, markers
        from ..map import series
        mode_classes = [
            RotateMouseMode,
            RotateSelectedMouseMode,
            TranslateMouseMode,
            TranslateSelectedMouseMode,
            ZoomMouseMode,
            ObjectIdMouseMode,
            SelectMouseMode,
            map.ContourLevelMouseMode,
            map.PlanesMouseMode,
            markers.MarkerMouseMode,
            markers.MarkCenterMouseMode,
            markers.ConnectMouseMode,
            series.PlaySeriesMouseMode,
        ]
        self._available_modes = [mode(session) for mode in mode_classes]

        self._mouse_modes = {}    # Maps 'left', 'middle', 'right', 'wheel', 'pause' to MouseMode instance

        # Mouse pause parameters
        self._last_mouse_time = None
        self._mouse_pause_interval = 0.5         # seconds
        self._mouse_pause_position = None

        self.bind_standard_mouse_modes()

        self._set_mouse_event_handlers()

    def _set_mouse_event_handlers(self):
        import wx
        c = self.graphics_window.opengl_canvas
        mouse_events = [
            (wx.EVT_LEFT_DOWN, "left", "mouse_down"),
            (wx.EVT_MIDDLE_DOWN, "middle", "mouse_down"),
            (wx.EVT_RIGHT_DOWN, "right", "mouse_down"),
            (wx.EVT_MOTION, None, "mouse_drag"),
            (wx.EVT_LEFT_UP, "left", "mouse_up"),
            (wx.EVT_MIDDLE_UP, "middle", "mouse_up"),
            (wx.EVT_RIGHT_UP, "right", "mouse_up"),
            (wx.EVT_LEFT_DCLICK, "left", "mouse_double"),
            (wx.EVT_MIDDLE_DCLICK, "middle", "mouse_double"),
            (wx.EVT_RIGHT_DCLICK, "right", "mouse_double"),
        ]
        for event, button, action in mouse_events:
            c.Bind(event, lambda e,b=button,a=action: self._dispatch_mouse_event(e,b,a))
        c.Bind(wx.EVT_MOUSEWHEEL, self._wheel_event)

    def _dispatch_mouse_event(self, event, button, action):
        canvas = self.graphics_window.opengl_canvas
        if action in ('mouse_down', 'mouse_double'):
            # remember button for later drag events
            if not canvas.HasCapture():
                canvas.CaptureMouse()
        elif action == 'mouse_drag':
            if not event.Dragging():
                return
        elif action == 'mouse_up':
            if canvas.HasCapture():
                canvas.ReleaseMouse()
        if button is None:
            if event.LeftIsDown():
                button = "middle" if event.AltDown() else "left"
            elif event.MiddleIsDown():
                button = "middle"
            elif event.RightIsDown():
                button = "right"
            else:
                return
            if not canvas.HasCapture():
                # a Windows thing; can lose mouse capture w/o mouse up
                return
        elif button == 'left' and event.AltDown():
            button = 'middle'
        m = self._mouse_modes.get(button)
        if m and hasattr(m, action):
            f = getattr(m, action)
            f(MouseEvent(event))

    def _wheel_event(self, event):
        f = self._mouse_modes.get('wheel')
        if f:
            f.wheel(MouseEvent(event))

    @property
    def modes(self):
        return self._available_modes

    @property
    def bindings(self):
        '''Maps button name to MouseMode object.'''
        return self._mouse_modes

    # Button is "left", "middle", "right", "wheel", or "pause"
    def bind_mouse_mode(self, button, mode):
        self._mouse_modes[button] = mode

    def bind_standard_mouse_modes(self, buttons = ('left', 'middle', 'right', 'wheel', 'pause')):
        modes = (
            ('left', 'rotate'),
            ('middle', 'translate'),
            ('right', 'zoom'),
            ('wheel', 'zoom'),
            ('pause', 'identify object'),
            )
        mmap = {m.name:m for m in self.modes}
        for button, mode_name in modes:
            if button in buttons:
                self.bind_mouse_mode(button, mmap[mode_name])

    def mouse_pause_tracking(self):
        '''Called periodically to check for mouse pause and invoke pause mode.'''
        cp = self._cursor_position()
        w,h = self.graphics_window.view.window_size
        x,y = cp
        if x < 0 or y < 0 or x >= w or y >= h:
            return      # Cursor outside of graphics window
        from time import time
        t = time()
        mp = self._mouse_pause_position
        if cp == mp:
            lt = self._last_mouse_time
            if lt and t >= lt + self._mouse_pause_interval:
                self._mouse_pause()
                self._mouse_pause_position = None
                self._last_mouse_time = None
            return
        self._mouse_pause_position = cp
        if mp:
            # Require mouse move before setting timer to avoid
            # repeated mouse pause callbacks at same point.
            self._last_mouse_time = t
            self._mouse_move_after_pause()

    def _cursor_position(self):
        import wx
        return self.graphics_window.ScreenToClient(wx.GetMousePosition())

    def _mouse_pause(self):
        m = self._mouse_modes.get('pause')
        if m:
            m.pause(self._mouse_pause_position)

    def _mouse_move_after_pause(self):
        m = self._mouse_modes.get('pause')
        if m:
            m.move_after_pause()

class MouseMode:

    name = 'mode name'
    icon_file = None

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

    def move_after_pause(self):
        pass

    def pixel_size(self, min_scene_frac = 1e-5):
        v = self.view
        psize = v.pixel_size()
        b = v.drawing_bounds()
        if not b is None:
            w = b.width()
            psize = max(psize, w*min_scene_frac)
        return psize

class SelectMouseMode(MouseMode):
    name = 'select'
    icon_file = 'select.png'

    def mouse_down(self, event):
        mouse_select(event, self.session, self.view)

def mouse_select(event, session, view):

    x,y = event.position()
    pick = view.first_intercept(x,y)
    toggle = event.shift_down()
    sel = session.selection
    if pick is None:
        if not toggle:
            sel.clear()
            session.logger.status('cleared selection')
    else:
        if not toggle:
            sel.clear()
        pick.select(toggle)
    sel.clear_hierarchy()

class RotateMouseMode(MouseMode):
    name = 'rotate'
    icon_file = 'rotate.png'

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
            mouse_select(event, self.session, self.view)
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

class RotateSelectedMouseMode(RotateMouseMode):
    name = 'rotate selected models'
    icon_file = 'rotate_h2o.png'

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
    name = 'translate'
    icon_file = 'translate.png'

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
    name = 'translate selected models'
    icon_file = 'move_h2o.png'

    def models(self):
        return top_selected(self.session)

class ZoomMouseMode(MouseMode):

    name = 'zoom'
    icon_file = 'zoom.png'

    def mouse_drag(self, event):        

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        self.zoom(3*psize*dy)

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        self.zoom(100*d*psize)

    def zoom(self, delta_z):
        v = self.view
        c = v.camera
        if c.name() == 'orthographic':
            c.field_width = max(c.field_width - delta_z, self.pixel_size())
            # TODO: Make camera field_width a property so it knows to redraw.
            c.redraw_needed = True
        else:
            shift = c.position.apply_without_translation((0, 0, delta_z))
            v.translate(shift)

class ObjectIdMouseMode(MouseMode):

    name = 'identify object'
    def pause(self, position):
        x,y = position
        p = self.view.first_intercept(x,y)

        # Show atom spec balloon
        pu = self.session.ui.main_window.graphics_window.popup
        if p:
            pu.show_text(p.description(), (x+10,y))
        else:
            pu.hide()

    def move_after_pause(self):
        # Hide atom spec balloon
        self.session.ui.main_window.graphics_window.popup.hide()

class MouseEvent:
    def __init__(self, event):
        self.event = event

    def shift_down(self):
        return self.event.ShiftDown()

    def position(self):
        return self.event.GetPosition()

    def wheel_value(self):
        return self.event.GetWheelRotation()/120.0   # Usually one wheel click is delta of 120
