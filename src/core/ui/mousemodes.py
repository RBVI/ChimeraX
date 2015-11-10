# vim: set expandtab ts=4 sw=4:

class MouseModes:

    def __init__(self, graphics_window, session):

        self.graphics_window = graphics_window
        self.session = session

        from .. import map, markers
        from ..map import series
        mode_classes = [
            SelectMouseMode,
            RotateMouseMode,
            TranslateMouseMode,
            ZoomMouseMode,
            RotateAndSelectMouseMode,
            TranslateSelectedMouseMode,
            RotateSelectedMouseMode,
            ClipMouseMode,
            ObjectIdMouseMode,
            map.ContourLevelMouseMode,
            map.PlanesMouseMode,
            markers.MarkerMouseMode,
            markers.MarkCenterMouseMode,
            markers.ConnectMouseMode,
            series.PlaySeriesMouseMode,
        ]
        self._available_modes = [mode(session) for mode in mode_classes]

        self._bindings = []  # List of MouseBinding

        
        import wx, sys
        if sys.platform == 'darwin':
            mod_bits = [(wx.MOD_ALT, 'alt'),
                        (wx.MOD_CONTROL, 'command'),		# On Mac, this is the Command key
                        (wx.MOD_RAW_CONTROL, 'control'),	# On Mac, ctrl.
                        (wx.MOD_SHIFT, 'shift')]
        else:
            mod_bits = [(wx.MOD_ALT, 'alt'),
                        (wx.MOD_CONTROL, 'control'),
                        (wx.MOD_SHIFT, 'shift')]
        self._modifier_bits = mod_bits

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
        if button is None and not canvas.HasCapture():
            # a Windows thing; can lose mouse capture w/o mouse up
            return

        button, modifiers = self._event_type(event, button)
        if button is None:
            return

        m = self.mode(button, modifiers)
        if m and hasattr(m, action):
            f = getattr(m, action)
            f(MouseEvent(event))

    def _event_type(self, event, button):
        import sys
        mac = (sys.platform == 'darwin')
        modifiers = self.key_modifiers(event)
        if button is None:
            # Drag event
            if event.LeftIsDown():
                button = "left"
            elif event.MiddleIsDown():
                button = "middle"
            elif event.RightIsDown():
                button = "right"
            else:
                button = None

        if button == 'left':
            if mac and 'command' in modifiers and not self._have_mode('left','command'):
                # Emulate right mouse on Mac
                button = 'right'
                modifiers.remove('command')
            elif mac and 'alt' in modifiers and not self._have_mode('left','alt'):
                # Emulate middle mouse on Mac
                button = 'middle'
                modifiers.remove('alt')
        elif button == 'right':
            if mac and 'control' in modifiers:
		# Mac wx ctrl-left is reported as ctrl-right.
                # No way to distinguish ctrl-left from ctrl-right,
                # so convert both to ctrl-left.
                button = 'left'

        return button, modifiers

    def _have_mode(self, button, modifier):
        for b in self.bindings:
            if b.exact_match(button, [modifier]):
                return True
        return False

    def _wheel_event(self, event):
        f = self.mode('wheel', self.key_modifiers(event))
        if f:
            f.wheel(MouseEvent(event))

    @property
    def modes(self):
        '''List of MouseMode instances.'''
        return self._available_modes

    @property
    def bindings(self):
        '''List of MouseBinding instances.'''
        return self._bindings

    def mode(self, button = 'left', modifiers = []):
        mb = [b for b in self.bindings if b.matches(button, modifiers)]
        if len(mb) == 1:
            m = mb[0].mode
        elif len(mb) > 1:
            m = max(mb, key = lambda b: len(b.modifiers)).mode
        else:
            m = None
        return m

    def bind_mouse_mode(self, button, modifiers, mode):
        '''
        Button is "left", "middle", "right", "wheel", or "pause".
        Modifiers is a list 0 or more of 'alt', 'command', 'control', 'shift'.
        Mode is a MouseMode instance.
        '''
        if button == 'right' and 'control' in modifiers:
            import sys
            mac = (sys.platform == 'darwin')
            if mac:
                self.session.logger.warning('Mac wx toolkit cannot distinguish ctrl-right mouse click '
                                            'from ctrl-left mouse click, so both are interpreted as '
                                            'ctrl-left mouse click.')
                return
        self.remove_binding(button, modifiers)
        if mode is not None:
            b = MouseBinding(button, modifiers, mode)
            self._bindings.append(b)

    def remove_binding(self, button, modifiers):
        self._bindings = [b for b in self.bindings if not b.exact_match(button, modifiers)]

    def bind_standard_mouse_modes(self, buttons = ('left', 'middle', 'right', 'wheel', 'pause')):
        standard_modes = (
            ('left', ['control'], 'select'),
            ('left', [], 'rotate'),
            ('middle', [], 'translate'),
            ('right', [], 'zoom'),
            ('wheel', [], 'zoom'),
            ('pause', [], 'identify object'),
            )
        mmap = {m.name:m for m in self.modes}
        for button, modifiers, mode_name in standard_modes:
            if button in buttons:
                self.bind_mouse_mode(button, modifiers, mmap[mode_name])

    def key_modifiers(self, event):
        mod = event.GetModifiers()
        modifiers = [mod_name for bit, mod_name in self._modifier_bits if bit & mod]
        return modifiers

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
        m = self.mode('pause')
        if m:
            m.pause(self._mouse_pause_position)

    def _mouse_move_after_pause(self):
        m = self.mode('pause')
        if m:
            m.move_after_pause()

class MouseBinding:
    def __init__(self, button, modifiers, mode):
        self.button = button		# 'left', 'middle', 'right', 'wheel', 'pause'
        self.modifiers = modifiers	# List of 'alt', 'command', 'control', 'shift'
        self.mode = mode		# MouseMode instance
    def matches(self, button, modifiers):
        return (button == self.button and
                len([k for k in self.modifiers if not k in modifiers]) == 0)
    def exact_match(self, button, modifiers):
        return button == self.button and set(modifiers) == set(self.modifiers)

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
    click_to_select = False

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
        if self.click_to_select:
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

class RotateAndSelectMouseMode(RotateMouseMode):
    name = 'rotate and select'
    icon_file = 'rotatesel.png'
    click_to_select = True

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

class ClipMouseMode(MouseMode):
    name = 'clip'
    icon_file = 'clip.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        ns,fs = {(False,False):(1,0),
                 (True,False):(1,1),
                 (False,True):(0,1),
                 (True,True):(1,-1)}[(event.shift_down(),event.alt_down())]
        self.clip_move((dx,-dy), ns, fs)

    def clip_move(self, delta_xy, near_shift, far_shift):

        v = self.view
        cam = v.camera
        clip = v.clip
        normal = clip.normal if clip.tilt else cam.view_direction()
        np, fp = clip.near_point, clip.far_point
        if np is None or fp is None:
            b = v.drawing_bounds()
            if b is None:
                return
            if np is None:
                np = b.center()
            if fp is None:
                fp = np + b.radius()*normal

        d = self.shift_distance(delta_xy, clip.tilt, cam, normal)
        snp = np + (near_shift*d)*normal
        sfp = fp + (far_shift*d)*normal

        from ..geometry import inner_product
        if inner_product(sfp-snp,normal) > 0:
            clip.near_point = snp
            clip.far_point = sfp
            clip.normal = normal
            v.redraw_needed = True

    def shift_distance(self, delta_xy, tilt, camera, normal):
        if tilt:
            # Measure drag direction along plane normal direction.
            nx,ny,nz = camera.position.inverse().apply_without_translation(normal)
            d = (nx*nx + ny*ny)
            if d > 0:
                nx /= d
                ny /= d
            else:
                nx = 0
                ny = 1
        else:
            # Vertical drag for face-on clipping
            nx = 0
            ny = 1
        dx,dy = delta_xy
        shift = (dx*nx + dy*ny) * self.pixel_size()
        return shift

class MouseEvent:
    def __init__(self, event):
        self.event = event

    def shift_down(self):
        return self.event.ShiftDown()

    def alt_down(self):
        return self.event.AltDown()

    def position(self):
        return self.event.GetPosition()

    def wheel_value(self):
        return self.event.GetWheelRotation()/120.0   # Usually one wheel click is delta of 120
