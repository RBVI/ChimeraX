# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Routines to setup OpenXR 3D screens such as Sony Spatial Reality
# or Acer SpatialLabs to handle the coordinate systems for these displays
# and mouse events and keyboard input.
#
def setup_openxr_screen(openxr_system_name, openxr_camera):
    if openxr_system_name == 'SonySRD System':
        _sony_spatial_reality_setup(openxr_camera)
    elif openxr_system_name == 'SpatialLabs Display Driver':
        _acer_spatial_labs_setup(openxr_camera)

def _sony_spatial_reality_setup(openxr_camera):
    # Flatpanel Sony Spatial Reality display with eye tracking.
    #   15.6" screen, 34 x 19 cm, tilted at 45 degree angle.
    # TODO: Distinguish 27" from 15.6" display.  Might use OpenXR vendorId
    from math import sqrt
    s2 = 1/sqrt(2)
    w,h = 0.34, 0.19	# Screen size meters
    from numpy import array
    screen_center = array((0, s2*h/2, -s2*h/2))
    from chimerax.geometry import rotation
    screen_orientation = rotation((1,0,0), -45)	# View direction 45 degree down.

    # Room size and center for view_all() positioning.
    c = openxr_camera
    c._initial_room_scene_size = h  # meters
    c._initial_room_center = screen_center

    # Make mouse zoom always perpendicular at screen center.
    # Sony rendered camera positions always are perpendicular
    # to screen but offset based on eye-tracking head position.
    # That leads to confusing skewed mouse zooming.
    c._desktop_view_point = screen_center

    # When leaving XR keep the same camera view point in the graphics window.
    c.keep_position = True

    # Set camera position and room to scene transform preserving
    # current camera view direction.
    v = c._session.main_view
    c.fit_view_to_room(room_width = w,
                       room_center = screen_center,
                       room_center_distance = 0.40,
                       screen_orientation = screen_orientation,
                       scene_center = v.center_of_rotation,
                       scene_camera = v.camera)

    _enable_xr_mouse_modes(c._session, openxr_window_captures_events = True)

def _acer_spatial_labs_setup(openxr_camera):
    # Flatpanel Acer SpatialLabs 27" display with eye tracking.
    w,h = 0.60, 0.34	# Screen size meters
    from numpy import array
    screen_center = array((0, 0, 0))
    from chimerax.geometry import identity
    screen_orientation = identity()

    # Room size and center for view_all() positioning.
    c = openxr_camera
    c._initial_room_scene_size = 0.7*h  # meters
    c._initial_room_center = screen_center

    # Make mouse zoom always perpendicular at screen center.
    # Sony rendered camera positions always are perpendicular
    # to screen but offset based on eye-tracking head position.
    # That leads to confusing skewed mouse zooming.
    c._desktop_view_point = screen_center

    # When leaving XR keep the same camera view point in the graphics window.
    c.keep_position = True

    # Set camera position and room to scene transform preserving
    # current camera view direction.
    v = c._session.main_view
    c.fit_view_to_room(room_width = w,
                       room_center = screen_center,
                       room_center_distance = 0.40,
                       screen_orientation = screen_orientation,
                       scene_center = v.center_of_rotation,
                       scene_camera = v.camera)

    _enable_xr_mouse_modes(c._session)

def _enable_xr_mouse_modes(session, screen_model_name = None,
                           openxr_window_captures_events = False):
    '''
    Allow mouse modes to work with mouse on Acer or Sony 3D displays.
    Both these displays create a fullscreen window. This mouse mode support
    works by creating a backing full-screen Qt window which receives the
     mouse events.
    '''
    screen = find_xr_screen(session, screen_model_name)
    if screen is None:
        return False
    XRBackingWindow(session, screen, in_front = openxr_window_captures_events)
    return True

xr_screen_model_names = ['ASV27-2P', 'SR Display']
def find_xr_screen(session, screen_model_name = None):
    model_names = [screen_model_name] if screen_model_name else xr_screen_model_names
    screens = session.ui.screens()
    for screen in screens:
        if screen.model() in model_names:
            return screen
    found_names = [screen.model() for screen in screens]
    msg = f'Could not find OpenXR screen {", ".join(model_names)} , only found {", ".join(found_names)}'
    session.logger.warning(msg)
    return None

class XRBackingWindow:
    '''
    Backing window for OpenXR autostereo 3D displays such as Acer SpatialLabs
    and Sony Spatial Reality to capture mouse and keyboard events when
    mouse is on the 3D display.
    '''
    def __init__(self, session, screen, in_front = False):
        self._session = session
        self._screen = screen
        
        # Create fullscreen backing Qt window on openxr screen.
        from Qt.QtWidgets import QWidget
        self._widget = w = QWidget()

        if in_front:
            self._make_transparent_in_front(w)
        
        w.move(screen.geometry().topLeft())
        w.showFullScreen()
        
        self._register_mouse_handlers()

        # Forward key press events
        w.keyPressEvent = session.ui.forward_keystroke

        # Remove backing window when openxr is turned off.
        session.triggers.add_handler('vr stopped', self._xr_quit)

        # Would be nice if mouse hover popup shown.
        # But we don't get any mouse hover events because the hover
        # code only triggers when the actual mouse cursor position
        # (from QCursor.pos()) is in the main graphics window.
        #   session.triggers.add_handler('mouse hover', self._mouse_hover)

    def _make_transparent_in_front(self, w):
        # On Sony Spatial Reality displays the full screen
        # window made by Sony OpenXR captures mouse events
        # so we instead put a transparent Qt window in front (July 2025).
        from Qt.QtCore import Qt
        w.setAttribute(Qt.WA_TranslucentBackground)
        w.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Unforunately the top level Qt translucent frameless window
        # also does not capture mouse events unless we add a frame
        # that has a tiny bit of opacity.
        from Qt.QtWidgets import QFrame, QVBoxLayout
        self._f = f = QFrame(w)
        f.setStyleSheet("background: rgba(2, 2, 2, 2);")

        # Make frame fill the entire parent window.
        layout = QVBoxLayout(w)
        w.setLayout(layout)
        layout.addWidget(f)

        # The following settings did not avoid the need to make
        # a child QFrame.
        #  w.setWindowFlags(Qt.FramelessWindowHint)
        #  w.setAttribute(Qt.WA_AlwaysStackOnTop)
        #  w.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        #  w.setStyleSheet("background:transparent;")
        #  w.setStyleSheet("background:green;")

    def _register_mouse_handlers(self):
        w = self._widget
        w.mousePressEvent = self._mouse_down
        w.mouseMoveEvent = self._mouse_drag
        w.mouseReleaseEvent = self._mouse_up
        w.mouseDoubleClickEvent = self._mouse_double_click
        w.wheelEvent = self._wheel

    def _mouse_down(self, event):
        self._dispatch_mouse_event(event, "mouse_down")
    def _mouse_drag(self, event):
        self._dispatch_mouse_event(event, "mouse_drag")
    def _mouse_up(self, event):
        self._dispatch_mouse_event(event, "mouse_up")
    def _mouse_double_click(self, event):
        self._dispatch_mouse_event(event, "mouse_double_click")
    def _wheel(self, event):
        self._dispatch_wheel_event(event)

    def _dispatch_mouse_event(self, event, action):
        '''
        Convert a mouse event from 3D screen coordinates to
        graphics pane coordinates and dispatch it.
        '''
        p = event.position()
        gx, gy = self._map_event_coordinates(p.x(), p.y())
        e = self._repositioned_event(event, gx, gy)
        mm = self._session.ui.mouse_modes
        mm._dispatch_mouse_event(e, action)

    def _dispatch_wheel_event(self, event):
        '''
        Convert a wheel event from 3D screen coordinates to
        graphics pane coordinates and dispatch it.
        '''
        p = event.position()
        gx, gy = self._map_event_coordinates(p.x(), p.y())
        e = self._repositioned_event(event, gx, gy)
        mm = self._session.ui.mouse_modes
        mm._wheel_event(e)

    def _map_event_coordinates(self, x, y):
        '''
        Handle different aspect ratio of 3D screen window and
        graphics window.  Graphics window has cropped version
        of 3D window image.
        '''
        w3d = self._widget
        w, h = w3d.width(), w3d.height()
        gw, gh = self._session.main_view.window_size
        if w == 0 or h == 0 or gw == 0 or gh == 0:
            return x, y
        fx,fy = x/w, y/h
        af = w*gh/(h*gw)
        if af > 1:
            afx = 0.5 + af * (fx - 0.5)
            afy = fy
        else:
            afx = fx
            afy = 0.5 + (1/af) * (fy - 0.5)
        gx, gy = afx * gw, afy * gh
        return gx, gy

    def _repositioned_event(self, event, x, y):
        from Qt.QtGui import QMouseEvent, QWheelEvent
        from Qt.QtCore import QPointF
        pos = QPointF(x, y)
        if isinstance(event, QMouseEvent):
            e = QMouseEvent(event.type(), pos, event.globalPosition(), event.button(), event.buttons(), event.modifiers(), event.device())
        elif isinstance(event, QWheelEvent):
            e = QWheelEvent(pos, event.globalPosition(), event.pixelDelta(), event.angleDelta(), event.buttons(), event.modifiers(), event.phase(), event.inverted(), device = event.device())
        else:
            raise RuntimeError(f'Event type is not mouse or wheel event {event}')
        return e

    def _xr_quit(self, *args):
        # Delete the backing window
        self._widget.deleteLater()
        self._widget = None
        return 'delete handler'

def _openxr_window(window_name = None):
    if window_name is None:
        window_name = 'Preview window Composited'  # For Sony SR 16" display
    handles = _find_window_handles_by_title(window_name)
    if len(handles) == 1:
        from Qt.QtGui import QWindow
        w = QWindow.fromWinId(handles[0])
        return w
    return None

def _find_window_handles_by_title(window_name):
    from win32 import win32gui

    def callback(hwnd, window_handles):
        if win32gui.GetWindowText(hwnd) == window_name:
            window_handles.append(hwnd)

    window_handles = []
    win32gui.EnumWindows(callback, window_handles)
    return window_handles
