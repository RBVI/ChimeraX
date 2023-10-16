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

'''
mouse_modes: Mouse modes
========================

Classes to create mouse modes and assign mouse buttons and modifier
keys to specific modes.
'''

class MouseMode:
    '''
    Classes derived from MouseMode implement specific mouse modes providing
    methods mouse_down(), mouse_up(), mouse_motion(), wheel(), pause() that
    are called when mouse events occur.  Which mouse button and modifier
    keys are detected by a mode is controlled by a different MauseModes class.
    '''

    name = 'mode name'
    '''
    Supported API.
    Name of the mouse mode used with the mousemode command.
    Should be unique among all mouse modes.
    '''

    icon_file = None
    '''
    Supported API.
    Image file name for an icon for this mouse mode to show in the mouse mode GUI panel.
    The icon file of this name needs to be in the mouse_modes tool icons subdirectory,
    should be PNG, square, and at least 64 pixels square.  It will be rescaled as needed.
    A none value means no icon will be shown in the gui interface.
    '''

    def __init__(self, session):
        '''Supported API.'''
        self.session = session
        self.view = session.main_view

        self.mouse_down_position = None
        '''Pixel position (x,y) of mouse-down, sometimes useful to detect on mouse-up
        whether any mouse motion occured. Set to None after mouse up.'''
        self.last_mouse_position = None
        '''Last mouse position during a mouse drag.'''
        self.double_click = False
        '''
        Supported API.
        Whether the last mouse-down was actually a double_click.  Can be used in the mouse-up
        event handler if different behavior needed after a double click.  There is a
        mouse_double_click method for doing something on a double click (which happens on the
        second mouse down), so this boolean is only for mouse-up handlers that behave differently
        after single vs. double clicks.'''

    def enable(self):
        '''
        Supported API.
        Called when mouse mode is enabled.
        Override if mode wants to know that it has been bound to a mouse button.
        '''
        pass

    def mouse_down(self, event):
        '''
        Supported API.
        Override this method to handle mouse down events.
        Derived methods can call this base class method to
        set mouse_down_position and last_mouse_position
        and properly handle double clicks.
        '''
        pos = event.position()
        self.mouse_down_position = pos
        self.last_mouse_position = pos
        self.double_click = False

    def mouse_drag(self, event):
        '''
        Supported API.
        Override this method to handle mouse drag events.
        '''
        pass

    def mouse_up(self, event):
        '''
        Supported API.
        Override this method to handle mouse down events.
        Derived methods can call this base class method to
        set mouse_down_position and last_mouse_position to None.
        '''
        self.mouse_down_position = None
        self.last_mouse_position = None

    def mouse_double_click(self, event):
        '''
        Supported API.
        Override this method to handle double clicks.
        Keep in mind that you will also receive the mouse_down and
        mouse_up events.  If your mouse_up handler needs to behave
        differently depending on whether it is the second part of a
        double click, have it check the self.double_click boolean,
        and make sure to call this base method so that the boolean is set.
        '''
        self.double_click = True

    def mouse_motion(self, event):
        '''
        Supported API.
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
        '''
        Supported API.
        Override this method to handle mouse wheel events.
        '''
        pass

    @property
    def uses_wheel(self):
        '''Return True if derived class implements the wheel() method.'''
        return getattr(self, 'wheel') != MouseMode.wheel

    def pause(self, position):
        '''
        Supported API.
        Override this method to take action when the mouse hovers for a time
        given by the MouseModes pause interval (default 0.5 seconds).
        '''
        pass

    def move_after_pause(self):
        '''
        Supported API.
        Override this method to take action when the mouse moves after a hover.
        This allows for instance undisplaying a popup help balloon window.
        '''
        pass

    def touchpad_two_finger_scale(self, event):
        '''
        Supported API.
        Override this method to take action when a two-finger pinching motion is
        used on a multitouch touchpad. The scale parameter is available as
        event.two_finger_scale and is a float, where values larger than 1
        indicate fingers moving apart and values less than 1 indicate fingers
        moving together.
        '''
        pass

    def touchpad_two_finger_twist(self, event):
        '''
        Supported API.
        Override this method to take action when a two-finger twisting motion
        is used on a multitouch touchpad. The angle parameter is available as
        event.two_finger_twist, and is a float representing the rotation
        angle in degrees.
        '''
        pass

    def touchpad_two_finger_trans(self, event):
        '''
        Supported API.
        Override this method to take action when a two-finger swiping motion is
        used on a multitouch touchpad. This method should use either
        event.two_finger_trans (a tuple of two floats delta_x and delta_y) or
        event.wheel_value (an effective mouse wheel value synthesized from
        delta_y). delta_x and delta_y are distances expressed as fractions of
        the total width of the trackpad.
        '''
        pass

    def touchpad_three_finger_trans(self, event):
        '''
        Supported API.
        Override this method to take action when a three-finger swiping motion
        is used on a multitouch touchpad. The move parameter is available as
        event.three_finger_trans and is a tuple of two floats: (delta_x,
        delta_y) representing the distance moved on the touchpad as a fraction
        of its width.
        '''
        pass

    def touchpad_four_finger_trans(self, event):
        '''
        Supported API.
        Override this method to take action when a four-finger swiping motion
        is used on a multitouch touchpad. The move parameter is available as
        event.three_finger_trans and is a tuple of two floats: (delta_x,
        delta_y) representing the distance moved on the touchpad as a fraction
        of its width.
        '''
        pass


    def pixel_size(self, center = None, min_scene_frac = 1e-5):
        '''
        Supported API.
        Report the pixel size in scene units at the center of rotation.
        Clamp the value to be at least min_scene_fraction times the width
        of the displayed models.
        '''
        v = self.view
        psize = v.pixel_size(center)
        b = v.drawing_bounds(cached_only = True)
        if not b is None:
            w = b.width()
            psize = max(psize, w*min_scene_frac)
        return psize

    @property
    def camera_position(self):
        c = self.view.camera
        # For multiview cameras like VR camera, use camera position for desktop window.
        if hasattr(c, 'desktop_camera_position'):
            cp = c.desktop_camera_position
            if cp is None:
                cp = c.position
        else:
            cp = c.position
        return cp

    @property
    def icon_path(self):
        return self.icon_location()

    @classmethod
    def icon_location(cls):
        file = cls.icon_file
        if file is None:
            return None

        from os import path
        if path.isabs(file):
            return file

        import inspect
        cfile = inspect.getfile(cls)
        p = path.join(path.dirname(cfile), file)
        return p

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





class MouseModes:
    '''
    Keep the list of available mouse modes and also which mode is bound
    to each mouse button (left, middle, right), or mouse button and modifier
    key (alt, command, control shift).
    The mouse modes object for a session is session.ui.mouse_modes
    '''
    def __init__(self, session):

        self.graphics_window = None
        self.session = session

        from .std_modes import standard_mouse_mode_classes
        self._available_modes = [mode(session) for mode in standard_mouse_mode_classes()]

        self._bindings = []  # List of MouseBinding instances
        self._trackpad_bindings = [] # List of MultitouchBinding instances

        from Qt.QtCore import Qt
        # Qt maps control to meta on Mac...

        # Mouse pause parameters
        self._last_mouse_time = None
        self._paused = False
        self._mouse_pause_interval = 0.5         # seconds
        self._mouse_pause_position = None

        session.triggers.add_trigger("set mouse mode")
        self.bind_standard_mouse_modes()
        self._last_mode = None			# Remember mode at mouse down and stay with it until mouse up

        from .trackpad import MultitouchTrackpad
        self.trackpad = MultitouchTrackpad(session, self)

    def bind_mouse_mode(self, mouse_button=None, mouse_modifiers=[], mode=None,
            trackpad_action=None, trackpad_modifiers=[]):
        '''
        Bind a MouseMode to a mouse click and/or a multitouch trackpad action
        with optional modifier keys.

        mouse_button is either None or one of ("left", "middle", "right", "wheel", or "pause").

        trackpad_action is either None or one of ("pinch", "twist", "two finger swipe",
        "three finger swipe" or "four finger swipe").

        mouse_modifiers and trackpad_modifiers are each a list of 0 or more of
        ("alt", "command", "control" or "shift").

        mode is a MouseMode instance.
        '''
        if mouse_button is not None:
            self._bind_mouse_mode(mouse_button, mouse_modifiers, mode)
        if trackpad_action is not None:
            self._bind_trackpad_mode(trackpad_action, trackpad_modifiers, mode)

    def _bind_mouse_mode(self, button, modifiers, mode):
        '''
        Button is "left", "middle", "right", "wheel", or "pause".
        Modifiers is a list 0 or more of 'alt', 'command', 'control', 'shift'.
        Mode is a MouseMode instance.
        '''
        self.remove_binding(button, modifiers)
        if mode is not None:
            from .std_modes import NullMouseMode
            if not isinstance(mode, NullMouseMode):
                b = MouseBinding(button, modifiers, mode)
                self._bindings.append(b)
                mode.enable()
            else:
                # make handling trigger simpler
                mode = None
        self.session.triggers.activate_trigger("set mouse mode", (button, modifiers, mode))

    def _bind_trackpad_mode(self, action, modifiers, mode):
        '''
        Action is one of ("pinch", "twist", "two finger swipe",
        "three finger swipe" or "four finger swipe"). Modifiers is a list of
        0 or more of ("alt", "command", "control" or "shift"). Mode is a
        MouseMode instance.
        '''
        self.remove_binding(trackpad_action=action, trackpad_modifiers=modifiers)
        if mode is not None:
            from .std_modes import NullMouseMode
            if not isinstance(mode, NullMouseMode):
                from .trackpad import MultitouchBinding
                b = MultitouchBinding(action, modifiers, mode)
                self._trackpad_bindings.append(b)
                mode.enable()

    def bind_standard_mouse_modes(self,
                                  buttons = ('left', 'middle', 'right', 'wheel', 'pause'),
                                  trackpad = ('two finger swipe', 'twist', 'pinch',
                                              'three finger swipe', 'four finger swipe')):

        '''
        Bind the standard mouse modes: left = rotate, ctrl-left = select, middle = translate,
        right = zoom, wheel = zoom, pause = identify object.
        '''
        standard_button_modes = (
            ('left', [], 'rotate'),
            ('left', ['control'], 'select'),
            ('middle', [], 'translate'),
            ('right', [], 'translate'),
            ('wheel', [], 'zoom'),
            ('pause', [], 'identify object'),
            )
        
        mmap = {m.name:m for m in self.modes}
        for button, modifiers, mode_name in standard_button_modes:
            if button in buttons:
                self.bind_mouse_mode(button, modifiers, mmap[mode_name])

        standard_trackpad_modes = (
            ('two finger swipe', [], 'rotate'),
            ('twist', [], 'rotate'),
            ('pinch', [], 'zoom'),
            ('three finger swipe', [], 'translate'),
            ('four finger swipe', [], 'swipe as scroll')
            )
                
        for trackpad_action, modifiers, mode_name in standard_trackpad_modes:
            self.bind_mouse_mode(trackpad_action=trackpad_action,
                                 trackpad_modifiers=modifiers, mode=mmap[mode_name])

    def add_mode(self, mode):
        '''Supported API. Add a MouseMode instance to the list of available modes.'''
        self._available_modes.append(mode)

    @property
    def bindings(self):
        '''List of MouseBinding instances.'''
        return self._bindings

    def mode(self, button = 'left', modifiers = [], exact = False):
        '''Return the MouseMode associated with a specified button and modifiers,
        or None if no mode is bound.'''
        if exact:
            mb = [b for b in self._bindings if b.exact_match(button, modifiers)]
        else:
            mb = [b for b in self._bindings if b.matches(button, modifiers)]
        if len(mb) == 1:
            m = mb[0].mode
        elif len(mb) > 1:
            m = max(mb, key = lambda b: len(b.modifiers)).mode
        else:
            m = None
        return m

    def trackpad_mode(self, action, modifiers=[], exact=False):
        '''
        Return the MouseMode associated with a specific multitouch action and
        modifiers, or None if no mode is bound.
        '''
        if exact:
            mb = [b for b in self._trackpad_bindings if b.exact_match(action, modifiers)]
        else:
            mb = [b for b in self._trackpad_bindings if b.matches(action, modifiers)]
        if len(mb) == 1:
            m = mb[0].mode
        elif len(mb) > 1:
            m = max(mb, key = lambda b: len(b.modifiers)).mode
        else:
            m = None
        return m

    @property
    def modes(self):
        '''List of MouseMode instances.'''
        return self._available_modes

    def named_mode(self, name):
        for m in self.modes:
            if m.name == name:
                return m
        return None

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
        if self._mouse_buttons_down():
            return
        from time import time
        t = time()
        moved = (cp != self._mouse_pause_position)
        if moved:
            self._mouse_pause_position = cp
            self._last_mouse_time = t
            if self._paused:
                # Moved after pausing
                self._paused = False
                self._mouse_move_after_pause()
        elif not self._paused:
            # Not moving but not paused for enough time yet.
            lt = self._last_mouse_time
            if lt and t >= lt + self._mouse_pause_interval:
                self._mouse_pause()
                self._paused = True

    def remove_binding(self, button=None, modifiers=[],
            trackpad_action=None, trackpad_modifiers=[]):
        '''
        Unbind the mouse button and modifier key combination.
        No mode will be associated with this button and modifier.
        '''
        if button is not None:
            self._bindings = [b for b in self.bindings if not b.exact_match(button, modifiers)]
        if trackpad_action is not None:
            self._trackpad_bindings = [b for b in self._trackpad_bindings if not b.exact_match(trackpad_action, trackpad_modifiers)]

    def remove_mode(self, mode):
        '''Remove a MouseMode instance from the list of available modes.'''
        self._available_modes.append(mode)
        self._bindings = [b for b in self.bindings if b.mode is not mode]

    def _cursor_position(self):
        from Qt.QtGui import QCursor
        gp = QCursor.pos()
        p = self.graphics_window.mapFromGlobal(gp)
        return p.x(), p.y()

    def _mouse_buttons_down(self):
        from Qt.QtCore import Qt
        return self.session.ui.mouseButtons() != Qt.MouseButton.NoButton

    def _dispatch_mouse_event(self, event, action):
        button, modifiers = self._event_type(event)
        if button is None:
            return

        if action == 'mouse_down':
            m = self.mode(button, modifiers)
            lm = self._last_mode
            if lm is not None and hasattr(lm, 'mouse_up'):
                # Another button was pressed so release current mouse mode.
                lm.mouse_up(MouseEvent(event, modifiers=modifiers))
            self._last_mode = m
            self.session.ui.dismiss_context_menu()	# Work around Qt 6.4 bug.
        else:
            m = self._last_mode	     # Stay with same mode until button up even if modifier keys change.
        if m and hasattr(m, action):
            f = getattr(m, action)
            f(MouseEvent(event, modifiers=modifiers))
        if action == 'mouse_up':
            self._last_mode = None

    def _event_type(self, event):
        modifiers = key_modifiers(event)

        # button() gives press/release buttons; buttons() gives move buttons
        from Qt.QtCore import Qt
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.LeftButton:
            button = 'left'
        elif b & Qt.MouseButton.MiddleButton:
            button = 'middle'
        elif b & Qt.MouseButton.RightButton:
            button = 'right'
        else:
            button = None

        # Mac-specific remappings...
        import sys
        if sys.platform == 'darwin':
            if button == 'left':
                # Emulate additional buttons for one-button mice/trackpads
                if 'command' in modifiers and not self._have_mode('left','command'):
                    button = 'right'
                    modifiers.remove('command')
                elif 'alt' in modifiers and not self._have_mode('left','alt'):
                    button = 'middle'
                    modifiers.remove('alt')
            elif button == 'right':
                # On the Mac, a control left-click comes back as a right-click
                # so map control-right to control-left.  We lose use of control-right,
                # but more important to have control-left!
                if 'control' in modifiers:
                    button = 'left'
        elif sys.platform == 'win32':
            # Emulate right mouse using Alt key for Windows trackpads
                if button == 'left':
                    if 'alt' in modifiers and not self._have_mode('left','alt'):
                        if 'control' in modifiers:
                            button = 'middle'
                            modifiers.remove('control')
                            modifiers.remove('alt')
                        else:
                            button = 'right'
                            modifiers.remove('alt')

        return button, modifiers

    def _dispatch_touch_event(self, touch_event):
        te = touch_event
        from .trackpad import touch_action_to_property
        for action, prop in touch_action_to_property.items():
            data = getattr(te, prop)
            if getattr(touch_event, prop) is None:
                continue
            m = self.trackpad_mode(action, te.modifiers)
            if m is not None:
                f = getattr(m, 'touchpad_'+prop)
                f(te)


        # t_string = ('Registered touch event: \n'
        #     'modifer keys pressed: {}\n'
        #     'wheel_value: {}\n'
        #     'two_finger_trans: {}\n'
        #     'two_finger_scale: {}\n'
        #     'two_finger_twist: {}\n'
        #     'three_finger_trans: {}\n'
        #     'four_finger_trans: {}').format(
        #         ', '.join(te._modifiers),
        #         te.wheel_value,
        #         te.two_finger_trans,
        #         te.two_finger_scale,
        #         te.two_finger_twist,
        #         te.three_finger_trans,
        #         te.four_finger_trans
        #     )
        # print(t_string)


    def _have_mode(self, button, modifier):
        for b in self.bindings:
            if b.exact_match(button, [modifier]):
                return True
        return False

    def _mouse_pause(self):
        m = self.mode('pause')
        if m:
            m.pause(self._mouse_pause_position)

    def _mouse_move_after_pause(self):
        m = self.mode('pause')
        if m:
            m.move_after_pause()

    def set_graphics_window(self, graphics_window):
        self.graphics_window = gw = graphics_window
        gw.mousePressEvent = lambda e, s=self: s._dispatch_mouse_event(e, "mouse_down")
        gw.mouseMoveEvent = lambda e, s=self: s._dispatch_mouse_event(e, "mouse_drag")
        gw.mouseReleaseEvent = lambda e, s=self: s._dispatch_mouse_event(e, "mouse_up")
        gw.mouseDoubleClickEvent = lambda e, s=self: s._dispatch_mouse_event(e, "mouse_double_click")
        gw.wheelEvent = self._wheel_event
        self.trackpad.set_graphics_window(gw)

    def _wheel_event(self, event):
        if self.trackpad.discard_trackpad_wheel_event(event):
            return	# Trackpad processing handled this event
        f = self.mode('wheel', key_modifiers(event))
        if f:
            f.wheel(MouseEvent(event))

class MouseEvent:
    '''
    Provides an interface to mouse event coordinates and modifier keys
    so that mouse modes do not directly depend on details of the window toolkit.
    '''
    def __init__(self, event = None, modifiers = None, position = None, wheel_value = None):
        self._event = event		# Window toolkit event object
        self._modifiers = modifiers	# List of 'shift', 'alt', 'control', 'command'
                                        # May differ from event modifiers when modifier used
                                        # for mouse button emulation.
        self._position = position	# x,y in pixels, can be None
        self._wheel_value = wheel_value # wheel clicks (usually 1 click equals 15 degrees rotation).

    def shift_down(self):
        '''
        Supported API.
        Does the mouse event have the shift key down.
        '''
        if self._modifiers is not None:
            return 'shift' in self._modifiers
        if self._event is not None:
            from Qt.QtCore import Qt
            return bool(self._event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        return False

    def ctrl_down(self):
        '''
        Supported API.
        Does the mouse event have the shift key down.
        '''
        if self._modifiers is not None:
            return 'control' in self._modifiers
        if self._event is not None:
            from Qt.QtCore import Qt
            return bool(self._event.modifiers() & Qt.KeyboardModifier.CtrlModifier)
        return False

    def alt_down(self):
        '''
        Supported API.
        Does the mouse event have the alt key down.
        '''
        if self._modifiers is not None:
            return 'alt' in self._modifiers
        if self._event is not None:
            from Qt.QtCore import Qt
            return bool(self._event.modifiers() & Qt.KeyboardModifier.AltModifier)
        return False

    def position(self):
        '''
        Supported API.
        Pair of floating point x,y pixel coordinates relative to upper-left corner of graphics window.
        These values can be fractional if pointer device gives subpixel resolution.
        '''
        if self._position is not None:
            return self._position
        e = self._event
        if e is not None:
            if hasattr(e, 'position'): # Qt 6
                p = e.position()
                return p.x(), p.y()
            elif hasattr(e, 'localPos'):	# QMouseEvent
                p = e.localPos()
                return p.x(), p.y()
            elif hasattr(e, 'posF'):	# QWheelEvent
                p = e.posF()
                return p.x(), p.y()
        return 0,0

    def global_position(self):
        e = self._event
        if e is not None:
            if hasattr(e, 'globalPosition'):
                p = e.globalPosition().toPoint()		# PyQt6
                return (p.x(), p.y())
            elif hasattr(e, 'globalPos'):
                p = e.globalPos()			# PyQt5
                return (p.x(), p.y())
        return (0,0)
    
    def wheel_value(self):
        '''
        Supported API.
        Number of clicks the mouse wheel was turned, signed float.
        One click is typically 15 degrees of wheel rotation.
        '''
        if self._wheel_value is not None:
            return self._wheel_value
        if self._event is not None:
            deltas = self._event.angleDelta()
            dx, dy = deltas.x(), deltas.y()
            delta = dy if abs(dy) > abs(dx) else dx
            return delta/120.0   # Usually one wheel click is delta of 120
        return 0

def mod_key_info(key_function):
    """Qt swaps control/meta on Mac, so centralize that knowledge here.
    The possible "key_functions" are: alt, control, command, and shift

    Returns the Qt modifier bit (e.g. Qt.KeyboardModifier.AltModifier) and name of the actual key
    """
    from Qt.QtCore import Qt
    mod = Qt.KeyboardModifier
    import sys
    if sys.platform == "win32" or sys.platform == "linux":
        command_name = "windows"
        alt_name = "alt"
    elif sys.platform == "darwin":
        command_name = "command"
        alt_name = "option"
    if key_function == "shift":
        return mod.ShiftModifier, "shift"
    elif key_function == "alt":
        return mod.AltModifier, alt_name
    elif key_function == "control":
        if sys.platform == "darwin":
            return mod.MetaModifier, command_name
        return mod.ControlModifier, "control"
    elif key_function == "command":
        if sys.platform == "darwin":
            return mod.ControlModifier, "control"
        return mod.MetaModifier, command_name

def key_modifiers(event):
    return decode_modifier_bits(event.modifiers())

_modifier_bits = None
def decode_modifier_bits(mod):
    global _modifier_bits
    if _modifier_bits is None:
        _function_keys = ["alt", "control", "command", "shift"]
        _modifier_bits = [(mod_key_info(fkey)[0], fkey) for fkey in _function_keys]
    modifiers = [mod_name for bit, mod_name in _modifier_bits if bit & mod]
    return modifiers


def keyboard_modifier_names(qt_keyboard_modifiers):
    from Qt.QtCore import Qt
    mod = Qt.KeyboardModifier
    import sys
    if sys.platform == 'darwin':
        modifiers = [(mod.ShiftModifier, 'shift'),
                     (mod.ControlModifier, 'command'),
                     (mod.AltModifier, 'option'),
                     (mod.AltModifier, 'alt'),
                     (mod.MetaModifier, 'control')]
    else:
        modifiers = [(mod.ShiftModifier, 'shift'),
                     (mod.ControlModifier, 'control'),
                     (mod.AltModifier, 'alt'),
                     (mod.MetaModifier, 'windows')]
    mnames = [mname for mflag, mname in modifiers if mflag & qt_keyboard_modifiers]
    return mnames
