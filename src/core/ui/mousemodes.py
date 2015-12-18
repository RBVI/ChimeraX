# vim: set expandtab ts=4 sw=4:

class MouseModes:
    '''
    Keep the list of available mouse modes and also which mode is bound
    to each mouse button (left, middle, right), or mouse button and modifier
    key (alt, command, control shift).
    The mouse modes object for a session is session.ui.main_window.graphics_window.mouse_modes
    '''
    def __init__(self, graphics_window, session):

        self.graphics_window = graphics_window
        self.session = session

        self._available_modes = [mode(session) for mode in standard_mouse_mode_classes()]

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
        modifiers = self._key_modifiers(event)
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
        f = self.mode('wheel', self._key_modifiers(event))
        if f:
            f.wheel(MouseEvent(event))

    @property
    def modes(self):
        '''List of MouseMode instances.'''
        return self._available_modes

    def add_mode(self, mode):
        '''Add a MouseMode instance to the list of available modes.'''
        self._available_modes.append(mode)

    def remove_mode(self, mode):
        '''Remove a MouseMode instance from the list of available modes.'''
        self._available_modes.append(mode)
        self._bindings = [b for b in self.bindings if b.mode is not mode]

    @property
    def bindings(self):
        '''List of MouseBinding instances.'''
        return self._bindings

    def mode(self, button = 'left', modifiers = []):
        '''Return the MouseMode associated with a specified button and modifiers,
        or None if no mode is bound.'''
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
        '''
        Unbind the mouse button and modifier key combination.
        No mode will be associated with this button and modifier.
        '''
        self._bindings = [b for b in self.bindings if not b.exact_match(button, modifiers)]

    def bind_standard_mouse_modes(self, buttons = ('left', 'middle', 'right', 'wheel', 'pause')):
        '''
        Bind the standard mouse modes: left = rotate, ctrl-left = select, middle = translate,
        right = zoom, wheel = zoom, pause = identify object.
        '''
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

    def _key_modifiers(self, event):
        mod = event.GetModifiers()
        modifiers = [mod_name for bit, mod_name in self._modifier_bits if bit & mod]
        return modifiers

    def mouse_pause_tracking(self):
        '''
        Called periodically to check for mouse pause and invoke pause mode.
        Typically this will be called by the redraw loop and is used to determine
        when a mouse pause occurs.
        '''
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
    '''
    Associates a mouse button ('left', 'middle', 'right', 'wheel', 'pause') and
    set of modifier keys ('alt', 'command', 'control', 'shift') with a MouseMode.
    '''
    def __init__(self, button, modifiers, mode):
        self.button = button		# 'left', 'middle', 'right', 'wheel', 'pause'
        self.modifiers = modifiers	# List of 'alt', 'command', 'control', 'shift'
        self.mode = mode		# MouseMode instance
    def matches(self, button, modifiers):
        '''
        Does this binding match the specified button and modifiers?
        A match requires all of the binding modifiers keys are among
        the specified modifiers (and possibly more).
        '''
        return (button == self.button and
                len([k for k in self.modifiers if not k in modifiers]) == 0)
    def exact_match(self, button, modifiers):
        '''
        Does this binding exactly match the specified button and modifiers?
        An exact match requires the binding modifiers keys are exactly the
        same set as the specified modifier keys.
        '''
        return button == self.button and set(modifiers) == set(self.modifiers)

class MouseMode:
    '''
    Classes derived from MouseMode implement specific mouse modes providing
    methods mouse_down(), mouse_up(), mouse_motion(), wheel(), pause() that
    are called when mouse events occur.  Which mouse button and modifier
    keys are detected by a mode is controlled by a different MauseModes class.
    '''

    name = 'mode name'
    '''
    Name of the mouse mode used with the mousemode command.
    Should be unique among all mouse modes.
    '''

    icon_file = None
    '''
    Image file name for an icon for this mouse mode to show in the mouse mode GUI panel.
    The icon file of this name needs to be in the mouse_modes tool icons subdirectory,
    should be PNG, square, and at least 64 pixels square.  It will be rescaled as needed.
    A none value means no icon will be shown in the gui interface.
    '''

    def __init__(self, session):
        self.session = session
        self.view = session.main_view

        self.mouse_down_position = None
        '''Pixel position (x,y) of mouse down, sometimes useful to detect on mouse up
        whether any mouse motion occured. Set to None after mouse up.'''
        self.last_mouse_position = None
        '''Last mouse position during a mouse drag.'''

    def mouse_down(self, event):
        '''
        Override this method to handle mouse down events.
        Derived methods can call this base class method to
        set mouse_down_position and last_mouse_position.
        '''
        pos = event.position()
        self.mouse_down_position = pos
        self.last_mouse_position = pos

    def mouse_up(self, event):
        '''
        Override this method to handle mouse down events.
        Derived methods can call this base class method to
        set mouse_down_position and last_mouse_position to None.
        '''
        self.mouse_down_position = None
        self.last_mouse_position = None

    def mouse_motion(self, event):
        '''
        Return the mouse motion in pixels (dx,dy) since the last mouse event.
        '''
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

    def wheel(self, event):
        '''Override this method to handle mouse wheel events.'''
        pass

    def pause(self, position):
        '''
        Override this method to take action when the mouse hovers for a time
        given by the MouseModes pause interval (default 0.5 seconds).
        '''
        pass

    def move_after_pause(self):
        '''
        Override this method to take action when the mouse moves after a hover.
        This allows for instance undisplaying a popup help balloon window.
        '''
        pass

    def pixel_size(self, min_scene_frac = 1e-5):
        '''
        Report the pixel size in scene units at the center of rotation.
        Clamp the value to be at least min_scene_fraction times the width
        of the displayed models.
        '''
        v = self.view
        psize = v.pixel_size()
        b = v.drawing_bounds()
        if not b is None:
            w = b.width()
            psize = max(psize, w*min_scene_frac)
        return psize

class SelectMouseMode(MouseMode):
    '''Mouse mode to select objects by clicking on them.'''
    name = 'select'
    icon_file = 'select.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

        self.minimum_drag_pixels = 5
        self.drag_color = (0,255,0,255)	# Green
        self._drawn_rectangle = None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        if self._is_drag(event):
            self._undraw_drag_rectangle()
            self._draw_drag_rectangle(event)

    def mouse_up(self, event):
        self._undraw_drag_rectangle()
        if self._is_drag(event):
            # Select objects in rectangle
            mouse_drag_select(self.mouse_down_position, event, self.session, self.view)
        else:
            # Select object under pointer
            mouse_select(event, self.session, self.view)
        MouseMode.mouse_up(self, event)

    def _is_drag(self, event):
        dp = self.mouse_down_position
        if dp is None:
            return False
        dx,dy = dp
        x, y = event.position()
        mp = self.minimum_drag_pixels
        return abs(x-dx) > mp or abs(y-dy) > mp

    def _draw_drag_rectangle(self, event):
        dx,dy = self.mouse_down_position
        x, y = event.position()
        v = self.session.main_view
        w,h = v.window_size
        v.draw_xor_rectangle(dx, h-dy, x, h-y, self.drag_color)
        self._drawn_rectangle = (dx,dy), (x,y)

    def _undraw_drag_rectangle(self):
        dr = self._drawn_rectangle
        if dr:
            (dx,dy), (x,y) = dr
            v = self.session.main_view
            w,h = v.window_size
            v.draw_xor_rectangle(dx, h-dy, x, h-y, self.drag_color)
            self._drawn_rectangle = None

def mouse_select(event, session, view):
    x,y = event.position()
    pick = view.first_intercept(x,y)
    toggle = event.shift_down()
    select_pick(session, pick, toggle)

def mouse_drag_select(start_xy, event, session, view):
    sx, sy = start_xy
    x,y = event.position()
    pick = view.rectangle_intercept(sx,sy,x,y)
    toggle = event.shift_down()
    select_pick(session, pick, toggle)

def select_pick(session, pick, toggle):
    sel = session.selection
    if pick is None:
        if not toggle:
            sel.clear()
            session.logger.status('cleared selection')
    else:
        if not toggle:
            sel.clear()
        if isinstance(pick, list):
            for p in pick:
                p.select(toggle)
        else:
            pick.select(toggle)
    sel.clear_promotion_history()

class RotateMouseMode(MouseMode):
    '''
    Mouse mode to rotate objects (actually the camera is moved) by dragging.
    Mouse drags initiated near the periphery of the window cause a screen z rotation,
    while other mouse drags use rotation axes lying in the plane of the screen and
    perpendicular to the direction of the drag.
    '''
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

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        self.rotate((0,1,0), 10*d)

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
    '''
    Mouse mode to rotate objects like RotateMouseMode.
    Also clicking without dragging selects objects.
    This mode allows click with no modifier keys to perform selection,
    while click and drag produces rotation.
    '''
    name = 'rotate and select'
    icon_file = 'rotatesel.png'
    click_to_select = True

class RotateSelectedMouseMode(RotateMouseMode):
    '''
    Mouse mode to rotate objects like RotateMouseMode but only selected
    models are rotated. Selected models are actually moved in scene
    coordinates instead of moving the camera. If nothing is selected,
    then the camera is moved as if all models are rotated.
    '''
    name = 'rotate selected models'
    icon_file = 'rotate_h2o.png'

    def models(self):
        return top_selected(self.session)

def top_selected(session):
    # Don't include parents of selected models.
    mlist = [m for m in session.selection.models()
             if ((len(m.child_models()) == 0 or m.selected or child_drawing_selected(m))
                 and not any_parent_selected(m))]
    return None if len(mlist) == 0 else mlist

def any_parent_selected(m):
    if not hasattr(m, 'parent') or m.parent is None:
        return False
    p = m.parent
    return p.selected or child_drawing_selected(p) or any_parent_selected(p)

def child_drawing_selected(m):
    # Check if a child is a Drawing and not a Model and is selected.
    from ..models import Model
    for d in m.child_drawings():
        if not isinstance(d, Model) and d.any_part_selected():
            return True
    return False

class TranslateMouseMode(MouseMode):
    '''
    Mouse mode to move objects in x and y (actually the camera is moved) by dragging.
    '''
    name = 'translate'
    icon_file = 'translate.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        self.translate((dx, -dy, 0))

    def wheel(self, event):
        d = event.wheel_value()
        self.translate((0,0,100*d))

    def translate(self, shift):

        psize = self.pixel_size()
        s = tuple(dx*psize for dx in shift)     # Scene units
        v = self.view
        step = v.camera.position.apply_without_translation(s)    # Scene coord system
        v.translate(step, self.models())

    def models(self):
        return None

class TranslateSelectedMouseMode(TranslateMouseMode):
    '''
    Mouse mode to move objects in x and y like TranslateMouseMode but only selected
    models are moved. Selected models are actually moved in scene
    coordinates instead of moving the camera. If nothing is selected,
    then the camera is moved as if all models are shifted.
    '''
    name = 'translate selected models'
    icon_file = 'move_h2o.png'

    def models(self):
        return top_selected(self.session)

class ZoomMouseMode(MouseMode):
    '''
    Mouse mode to move objects in z, actually the camera is moved
    and the objects remain at their same scene coordinates.
    '''
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
    '''
    Mouse mode to that shows the name of an object in a popup window
    when the mouse is hovered over the object for 0.5 seconds.
    '''
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

class NullMouseMode(MouseMode):
    '''Used to assign no mode to a mouse button.'''
    name = 'none'

class ClipMouseMode(MouseMode):
    '''
    Move clip planes.
    Move front plane with no modifiers, back plane with alt,
    both planes with shift, and slab thickness with alt and shift.
    Move scene planes unless only near/far planes are enabled.
    '''
    name = 'clip'
    icon_file = 'clip.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        shift, alt = event.shift_down(), event.alt_down()
        front_shift = 1 if shift or not alt else 0
        back_shift = 0 if not (alt or shift) else (1 if alt and shift else -1)
        self.clip_move((dx,-dy), front_shift, back_shift)

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        self.clip_move(None, 1, 0, delta = 10*psize*d)

    def clip_move(self, delta_xy, front_shift, back_shift, delta = None):
        pf, pb = self._planes(front_shift, back_shift)
        if pf is None and pb is None:
            return

        p = pf or pb
        if delta is not None:
            d = delta
        elif p and p.camera_normal is None:
            # Move scene clip plane
            d = self._tilt_shift(delta_xy, self.view.camera, p.normal)
        else:

            # near/far clip
            d = delta_xy[1]*self.pixel_size()

        # Check if slab thickness becomes less than zero.
        dt = -d*(front_shift+back_shift)
        if pf and pb and dt < 0:
            from ..geometry import inner_product
            sep = inner_product(pb.plane_point - pf.plane_point, pf.normal)
            if sep + dt <= 0:
                # Would make slab thickness less than zero.
                return

        if pf:
            pf.plane_point = pf.plane_point + front_shift*d*pf.normal
        if pb:
            pb.plane_point = pb.plane_point + back_shift*d*pb.normal

    def _planes(self, front_shift, back_shift):
        v = self.view
        p = v.clip_planes
        pfname, pbname = (('front','back') if p.find_plane('front') or p.find_plane('back') or not p.planes() 
                          else ('near','far'))
        
        pf, pb = p.find_plane(pfname), p.find_plane(pbname)
        from ..commands.clip import adjust_plane
        c = v.camera
        cfn, cbn = ((0,0,-1), (0,0,1)) if pfname == 'near' else (None, None)

        if front_shift and pf is None:
            b = v.drawing_bounds()
            if pb:
                offset = -1 if b is None else -0.2*b.radius()
                pf = adjust_plane(pfname, offset, pb.plane_point, -pb.normal, p, v, cfn)
            elif b:
                normal = v.camera.view_direction()
                offset = 0
                pf = adjust_plane(pfname, offset, b.center(), normal, p, v, cfn)

        if back_shift and pb is None:
            b = v.drawing_bounds()
            offset = -1 if b is None else -0.2*b.radius()
            if pf:
                pb = adjust_plane(pbname, offset, pf.plane_point, -pf.normal, p, v, cbn)
            elif b:
                normal = -v.camera.view_direction()
                pb = adjust_plane(pbname, offset, b.center(), normal, p, v, cbn)

        return pf, pb

    def _tilt_shift(self, delta_xy, camera, normal):
        # Measure drag direction along plane normal direction.
        nx,ny,nz = camera.position.inverse().apply_without_translation(normal)
        from math import sqrt
        d = sqrt(nx*nx + ny*ny)
        if d > 0:
            nx /= d
            ny /= d
        else:
            nx = 0
            ny = 1
        dx,dy = delta_xy
        shift = (dx*nx + dy*ny) * self.pixel_size()
        return shift

class ClipRotateMouseMode(MouseMode):
    '''
    Rotate clip planes.
    '''
    name = 'clip rotate'
    icon_file = 'cliprot.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        axis, angle = self._drag_axis_angle(dx, dy)
        self.clip_rotate(axis, angle)

    def _drag_axis_angle(self, dx, dy):
        '''Axis in camera coords, angle in degrees.'''
        from math import sqrt
        d = sqrt(dx*dx + dy*dy)
        axis = (dy/d, dx/d, 0) if d > 0 else (0,1,0)
        angle = d
        return axis, angle

    def wheel(self, event):
        d = event.wheel_value()
        self.clip_rotate(axis = (0,1,0), angle = 10*d)

    def clip_rotate(self, axis, angle):
        v = self.view
        scene_axis = v.camera.position.apply_without_translation(axis)
        from ..geometry import rotation
        r = rotation(scene_axis, angle, v.center_of_rotation)
        for p in self._planes():
            p.normal = r.apply_without_translation(p.normal)
            p.plane_point = r * p.plane_point

    def _planes(self):
        v = self.view
        cp = v.clip_planes
        rplanes = [p for p in cp.planes() if p.camera_normal is None]
        if len(rplanes) == 0:
            from ..commands.clip import adjust_plane
            pn, pf = cp.find_plane('near'), cp.find_plane('far')
            if pn is None and pf is None:
                # Create clip plane since none are enabled.
                b = v.drawing_bounds()
                p = adjust_plane('front', 0, b.center(), v.camera.view_direction(), cp)
                rplanes = [p]
            else:
                # Convert near/far clip planes to scene planes.
                if pn:
                    rplanes.append(adjust_plane('front', 0, pn.plane_point, pn.normal, cp))
                    cp.remove_plane('near')
                if pf:
                    rplanes.append(adjust_plane('back', 0, pf.plane_point, pf.normal, cp))
                    cp.remove_plane('far')
        return rplanes


class MouseEvent:
    '''
    Provides an interface to mouse event coordinates and modifier keys
    so that mouse modes do not directly depend on details of the window toolkit.
    '''
    def __init__(self, event):
        self._event = event	# Window toolkit event object

    def shift_down(self):
        '''Does the mouse event have the shift key down.'''
        return self._event.ShiftDown()

    def alt_down(self):
        '''Does the mouse event have the alt key down.'''
        return self._event.AltDown()

    def position(self):
        '''Pair of integer x,y pixel coordinates relative to upper-left corner of graphics window.'''
        return self._event.GetPosition()

    def wheel_value(self):
        '''
        Number of clicks the mouse wheel was turned, signed float.
        One click is typically 15 degrees of wheel rotation.
        '''
        return self._event.GetWheelRotation()/120.0   # Usually one wheel click is delta of 120

def standard_mouse_mode_classes():
    '''List of core MouseMode classes.'''
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
        ClipRotateMouseMode,
        ObjectIdMouseMode,
        map.ContourLevelMouseMode,
        map.PlanesMouseMode,
        markers.MarkerMouseMode,
        markers.MarkCenterMouseMode,
        markers.ConnectMouseMode,
        series.PlaySeriesMouseMode,
        NullMouseMode,
    ]
    return mode_classes
