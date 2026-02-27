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
# Routines to setup OpenXR 3D screens such as Sony Spatial Reality,
# Acer SpatialLabs, or Samsung Odyssey 3D (via SteamVR vrto3d driver)
# to handle the coordinate systems for these displays
# and mouse events and keyboard input.
#
def setup_openxr_screen(openxr_system_name, openxr_camera):
    if openxr_system_name == 'SonySRD System':
        _sony_spatial_reality_setup(openxr_camera)
    elif openxr_system_name == 'SpatialLabs Display Driver':
        _acer_spatial_labs_setup(openxr_camera)
    elif 'vrto3d' in openxr_system_name.lower():
        _vrto3d_screen_setup(openxr_camera)

def _sony_spatial_reality_setup(openxr_camera):
    # Flatpanel Sony Spatial Reality display with eye tracking.
    #   15.6" screen, 34 x 19 cm, tilted at 45 degree angle, screen name "SR Display"
    #   27" screen, 58 x 33 cm, screen name "SR Display GB"
    screen = find_xr_screen(openxr_camera._session)
    if screen is None or screen.model() == 'SR Display GB':
        # 27" display
        # Unknown why it needs a scale factor of 9.  Determined by Utz Ermel.
        scale = 9
        w,h = 0.58*scale, 0.33*scale
    else:
        w,h = 0.34, 0.19	# Screen size meters

    from math import sqrt
    s2 = 1/sqrt(2)
    from numpy import array
    screen_center = array((0, s2*h/2, -s2*h/2))
    from chimerax.geometry import rotation
    screen_orientation = rotation((1,0,0), -45)	# View direction 45 degree down.
    # Center model behind screen for more comfortable viewing.
    model_center = screen_center + (h/4) * array((0, -s2, -s2))

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
                       room_center = model_center,
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
    model_center = screen_center + h/4 * array((0,0,-1))

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
                       room_center = model_center,
                       room_center_distance = 0.40,
                       screen_orientation = screen_orientation,
                       scene_center = v.center_of_rotation,
                       scene_camera = v.camera)

    _enable_xr_mouse_modes(c._session)

def _vrto3d_screen_setup(openxr_camera):
    # SteamVR vrto3d driver used with autostereo 3D displays such as
    # Samsung Odyssey 3D (flat vertical panel with eye tracking).
    # vrto3d emulates a VR headset via SteamVR so that OpenXR apps
    # produce stereo output which vrto3d converts to SBS for the
    # display's lenticular lens and eye tracking.
    #
    # Unlike Sony/Acer which use native OpenXR screen drivers,
    # vrto3d goes through SteamVR which handles room positioning.
    # We only need to enable mouse modes here -- no fit_view_to_room().
    #
    # direct_pick: vrto3d per-eye render is portrait (e.g. 1920x2160)
    # while the screen is landscape. The standard coordinate mapping
    # through the graphics pane loses accuracy due to aspect ratio
    # mismatch. direct_pick maps backing window coordinates directly
    # to the XR render texture, bypassing the graphics pane.
    _enable_xr_mouse_modes(openxr_camera._session,
                           openxr_window_captures_events = True,
                           direct_pick = True,
                           cursor_3d = True)

def _enable_xr_mouse_modes(session, screen_model_name = None,
                           openxr_window_captures_events = False,
                           direct_pick = False,
                           cursor_3d = False):
    '''
    Allow mouse modes to work with mouse on Acer, Sony, or Samsung 3D displays.
    These displays create a fullscreen window. This mouse mode support
    works by creating a backing full-screen Qt window which receives the
    mouse events.
    '''
    screen = find_xr_screen(session, screen_model_name)
    if screen is None:
        session.logger.warning('Could not enable mouse on OpenXR screen.')
        return False
    XRBackingWindow(session, screen, in_front = openxr_window_captures_events,
                    direct_pick = direct_pick, cursor_3d = cursor_3d)
    session.logger.info(f'Enabled mouse on OpenXR screen "{screen.model()}"')
    return True

xr_screen_model_names = ['ASV27-2P', '1ASV27-2P', 'DS1_156', 'SR Display', 'SR Display GB',
                         'Odyssey G90XF', 'Odyssey G90XH']
def find_xr_screen(session, screen_model_name = None):
    model_names = [screen_model_name] if screen_model_name else xr_screen_model_names
    screens = session.ui.screens()
    for screen in screens:
        if screen.model() in model_names:
            return screen
    found_names = [screen.model() for screen in screens]
    msg = f'Could not find OpenXR screen, found screens {", ".join(found_names)} which do not match any OpenXR screen names understood by ChimeraX: {", ".join(model_names)}.'
    session.logger.warning(msg)
    return None

_CURSOR_STYLES = ('sphere', 'crosshair', 'diamond', 'arrow', 'pointer')

def _crosshair_geometry(size):
    '''Three orthogonal thin rectangular prisms forming a 3D crosshair.'''
    import numpy as np
    arm = size
    t = size * 0.1  # Half-thickness of each arm
    verts = []
    norms = []
    tris = []
    for axis in range(3):
        ext = [t, t, t]
        ext[axis] = arm
        hx, hy, hz = ext
        o = len(verts)
        v = [[-hx,-hy,-hz], [hx,-hy,-hz], [hx,hy,-hz], [-hx,hy,-hz],
             [-hx,-hy, hz], [hx,-hy, hz], [hx,hy, hz], [-hx,hy, hz]]
        verts.extend(v)
        for p in v:
            n = np.array(p, dtype=np.float32)
            mag = np.linalg.norm(n)
            norms.append(n / mag if mag > 0 else np.array([0,1,0], dtype=np.float32))
        tris.extend([
            [o+0,o+2,o+1], [o+0,o+3,o+2],
            [o+4,o+5,o+6], [o+4,o+6,o+7],
            [o+0,o+1,o+5], [o+0,o+5,o+4],
            [o+2,o+3,o+7], [o+2,o+7,o+6],
            [o+0,o+4,o+7], [o+0,o+7,o+3],
            [o+1,o+2,o+6], [o+1,o+6,o+5],
        ])
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(tris, np.int32)

def _diamond_geometry(size):
    '''Octahedron (diamond) shape.'''
    import numpy as np
    s = size
    verts = np.array([
        [ s, 0, 0], [-s, 0, 0],
        [ 0, s, 0], [ 0,-s, 0],
        [ 0, 0, s], [ 0, 0,-s],
    ], dtype=np.float32)
    norms = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
    ], dtype=np.float32)
    tris = np.array([
        [0,2,4], [2,1,4], [1,3,4], [3,0,4],
        [2,0,5], [1,2,5], [3,1,5], [0,3,5],
    ], dtype=np.int32)
    return verts, norms, tris

def _arrow_geometry(size):
    '''3D cone pointer — tip at origin, base extends in +Y.
    Tip sits on the surface; base points toward the camera.'''
    import numpy as np
    length = size * 5     # Cone length (visible in stereo)
    base_r = size * 0.6   # Base radius
    n_seg = 16
    # Tip at origin
    verts = [[0, 0, 0]]
    norms = [[0, -1, 0]]  # Tip normal points down
    # Base circle at +length (away from surface, toward camera)
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    slope = base_r / length
    for a in angles:
        ca, sa = np.cos(a), np.sin(a)
        verts.append([base_r * ca, length, base_r * sa])
        nx, nz = ca, sa
        ny = -slope
        mag = np.sqrt(nx*nx + ny*ny + nz*nz)
        norms.append([nx/mag, ny/mag, nz/mag])
    # Base center cap
    bc = len(verts)
    verts.append([0, length, 0])
    norms.append([0, 1, 0])
    # Cone surface triangles
    tris = []
    for i in range(n_seg):
        tris.append([0, 1 + (i + 1) % n_seg, 1 + i])
    # Base cap triangles
    for i in range(n_seg):
        tris.append([bc, 1 + i, 1 + (i + 1) % n_seg])
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(tris, np.int32)

def _pointer_geometry(size):
    '''Classic mouse pointer cursor, extruded with side walls for
    real 3D depth.  Tip at origin so it sits on the surface;
    body extends in -Y (toward camera after rotation).'''
    import numpy as np
    s = size
    t = s * 0.15  # Half-thickness for stereo parallax
    # Classic arrow cursor silhouette in XY plane, tip at origin
    tip_y = s * 1.2  # original tip offset, used to centre on tip
    pts = [
        [0, 0],                          # 0: tip (at origin)
        [-s*0.40, -s*0.05 - tip_y],      # 1: left wing
        [-s*0.12,  s*0.15 - tip_y],      # 2: left notch
        [-s*0.12, -s*0.55 - tip_y],      # 3: tail bottom-left
        [ s*0.12, -s*0.55 - tip_y],      # 4: tail bottom-right
        [ s*0.12,  s*0.15 - tip_y],      # 5: right notch
        [ s*0.40, -s*0.05 - tip_y],      # 6: right wing
    ]
    faces_2d = [
        [0, 1, 2], [0, 2, 5], [0, 5, 6],  # arrowhead
        [2, 3, 4], [2, 4, 5],              # tail shaft
    ]
    verts = []
    norms = []
    tris = []
    n = len(pts)
    # Front face (+z)
    for p in pts:
        verts.append([p[0], p[1], t])
        norms.append([0, 0, 1])
    for f in faces_2d:
        tris.append(f)
    # Back face (-z, reversed winding)
    for p in pts:
        verts.append([p[0], p[1], -t])
        norms.append([0, 0, -1])
    for f in faces_2d:
        tris.append([n + f[0], n + f[2], n + f[1]])
    # Side walls (connect front and back outline edges)
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0)]
    base = 2 * n
    for i, (a, b) in enumerate(edges):
        pa, pb = pts[a], pts[b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        # Outward normal perpendicular to edge (CCW polygon)
        nx, ny = dy, -dx
        mag = np.sqrt(nx*nx + ny*ny)
        if mag > 0:
            nx, ny = nx/mag, ny/mag
        vi = base + i * 4
        verts.extend([
            [pa[0], pa[1],  t],
            [pb[0], pb[1],  t],
            [pb[0], pb[1], -t],
            [pa[0], pa[1], -t],
        ])
        norms.extend([[nx, ny, 0]] * 4)
        tris.append([vi, vi+1, vi+2])
        tris.append([vi, vi+2, vi+3])
    return np.array(verts, np.float32), np.array(norms, np.float32), np.array(tris, np.int32)

def _view_rotation(camera):
    '''Return 3x3 rotation whose columns are camera right / up / -forward
    expressed in scene coordinates.  This is the TRANSPOSE of the view
    matrix rotation (view maps scene→camera; we need camera→scene).
    Screen-fixed overlays use this so they do not rotate with the molecule.'''
    try:
        vp = camera.view(camera.position, 0)   # left eye view transform
        return vp.zero_translation().remove_scale().axes().T.copy()
    except Exception:
        return camera.position.zero_translation().remove_scale().axes().T.copy()

def _arrow_view_rotation(R):
    '''Modify view rotation so arrow cone tip points diagonally into
    screen (between camera forward and down).  Cone geometry has tip
    at model -Y, so we remap model Y to point away from the tip.'''
    import numpy as np
    from numpy.linalg import norm
    cam_fwd = -R[:, 2]
    cam_down = -R[:, 1]
    tip_dir = cam_fwd + cam_down
    td_len = norm(tip_dir)
    tip_dir = tip_dir / td_len if td_len > 0 else cam_fwd
    y_ax = -tip_dir
    x_ax = R[:, 0].copy()
    z_ax = np.cross(x_ax, y_ax)
    zl = norm(z_ax)
    if zl > 0:
        z_ax /= zl
    x_ax = np.cross(y_ax, z_ax)
    return np.column_stack([x_ax, y_ax, z_ax])

def _pointer_view_rotation(R):
    '''Tilt pointer into screen and lean sideways like a real cursor.
    1) Lean 20° clockwise (tip toward upper-left, classic cursor pose)
    2) Tilt 25° into screen so the face is partly visible.'''
    import numpy as np
    import math
    from numpy.linalg import norm
    right = R[:, 0].copy()
    up = R[:, 1].copy()
    fwd = -R[:, 2]  # camera forward (into screen)
    # 1) Lean sideways: rotate around forward axis, clockwise from
    #    camera POV so tip goes upper-left like a real mouse cursor.
    lean = math.radians(-20)
    cl, sl = math.cos(lean), math.sin(lean)
    right2 = cl * right + sl * up
    up2 = -sl * right + cl * up
    # 2) Tilt into screen: rotate around the leaned right axis.
    tilt = math.radians(25)
    ct, st = math.cos(tilt), math.sin(tilt)
    y_ax = ct * up2 + st * fwd
    x_ax = right2.copy()
    # Orthonormalise
    z_ax = np.cross(x_ax, y_ax)
    zl = norm(z_ax)
    if zl > 0:
        z_ax /= zl
    x_ax = np.cross(y_ax, z_ax)
    return np.column_stack([x_ax, y_ax, z_ax])


class Cursor3D:
    '''
    3D cursor for autostereo displays. Renders a small shape in the
    scene at the depth of whatever is under the mouse, so it appears
    at the correct stereo depth instead of being flat on the screen.
    Press C to cycle styles: sphere, crosshair, diamond, arrow, pointer.

    Orientation is baked into vertex positions each frame via
    set_geometry() so it stays screen-fixed in XR rendering.
    model.position handles translation only.
    '''
    def __init__(self, session, style = 'sphere', radius = 0.4):
        self._session = session
        self._radius = radius
        self._style = style
        from chimerax.core.models import Surface
        self._model = m = Surface('3D Cursor', session)
        m.color = (255, 150, 0, 180)  # Orange, semi-transparent
        m.pickable = False
        m.display = False
        self._apply_style(style)
        session.models.add([m])

    @property
    def style(self):
        return self._style

    def set_style(self, style):
        self._style = style
        self._apply_style(style)

    def _apply_style(self, style):
        import numpy as np
        r = self._radius
        if style == 'sphere':
            from chimerax.surface import sphere_geometry2
            va, na, ta = sphere_geometry2(80)
            va = r * va
        elif style == 'crosshair':
            va, na, ta = _crosshair_geometry(r)
        elif style == 'diamond':
            va, na, ta = _diamond_geometry(r)
        elif style == 'arrow':
            va, na, ta = _arrow_geometry(r)
        elif style == 'pointer':
            va, na, ta = _pointer_geometry(r * 2.5)
        else:
            return
        # Store base geometry as float64 for precision when rotating.
        # Rotation is baked into vertices each frame via set_geometry()
        # rather than using model.position rotation, which gave incorrect
        # orientation in the XR rendering pipeline.
        self._base_va = np.array(va, dtype=np.float64)
        self._base_na = np.array(na, dtype=np.float64)
        self._base_ta = np.array(ta, dtype=np.int32)
        self._model.set_geometry(
            np.array(va, dtype=np.float32),
            np.array(na, dtype=np.float32),
            self._base_ta)

    def update(self, x, y):
        view = self._session.main_view
        pick = view.picked_object(int(x), int(y))
        pos = None
        if pick is not None and hasattr(pick, 'position') and pick.position is not None:
            pos = pick.position
            # Offset toward camera so cursor doesn't z-fight with
            # the surface.
            cam_origin = view.camera.position.origin()
            from numpy.linalg import norm
            to_cam = cam_origin - pos
            dist = norm(to_cam)
            if dist > 0:
                offset = 0.3
                pos = pos + offset * (to_cam / dist)
        else:
            # Nothing under cursor -- place along camera ray at scene
            # depth, slightly closer to viewer for stereo pop-out.
            cam = view.camera
            origin, direction = cam.ray(int(x), int(y), view.window_size)
            if origin is not None:
                cofr = view.center_of_rotation
                from numpy.linalg import norm
                dist = norm(cofr - cam.position.origin())
                pos = origin + dist * 0.97 * direction

        if pos is not None:
            self._place_geometry(pos, view.camera)
            self._model.display = True
        else:
            self._model.display = False

    def _place_geometry(self, pos, camera):
        '''Place cursor at pos with screen-fixed orientation.
        Rotation is baked into vertex positions via set_geometry();
        model.position handles translation only.'''
        import numpy as np
        from chimerax.geometry import Place
        if self._style == 'sphere':
            va = self._base_va.astype(np.float32)
            na = self._base_na.astype(np.float32)
        else:
            R = _view_rotation(camera)
            if self._style == 'arrow':
                R = _arrow_view_rotation(R)
            elif self._style == 'pointer':
                R = _pointer_view_rotation(R)
            va = (R @ self._base_va.T).T.astype(np.float32)
            na = (R @ self._base_na.T).T.astype(np.float32)
        self._model.set_geometry(va, na, self._base_ta)
        self._model.position = Place(origin=pos)

    def hide(self):
        if self._model is not None:
            self._model.display = False

    def delete(self):
        if self._model is not None:
            self._session.models.remove([self._model])
            self._model = None

class SelectionRect3D:
    '''3D selection rectangle for autostereo displays.
    Renders a semi-transparent quad in the scene at the center-of-rotation
    depth so the ctrl-drag selection area is visible in stereo.'''
    def __init__(self, session):
        self._session = session
        from chimerax.core.models import Surface
        self._model = m = Surface('3D Selection', session)
        m.color = (100, 180, 255, 60)  # Light blue, semi-transparent
        m.pickable = False
        m.display = False
        m.use_lighting = False
        session.models.add([m])

    def update(self, x0, y0, x1, y1):
        '''Update rectangle corners from graphics coordinates.'''
        import numpy as np
        view = self._session.main_view
        cam = view.camera
        cofr = view.center_of_rotation
        # Plane perpendicular to view direction at center of rotation.
        # Use camera.view() for the actual rendering view direction
        # (handles XR room_to_scene scale and rotation correctly).
        view_dir = -_view_rotation(cam)[:, 2]
        ws = view.window_size
        corners_3d = []
        for cx, cy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            origin, direction = cam.ray(int(cx), int(cy), ws)
            if origin is None:
                self._model.display = False
                return
            denom = np.dot(direction, view_dir)
            if abs(denom) < 1e-10:
                self._model.display = False
                return
            t = np.dot(cofr - origin, view_dir) / denom
            corners_3d.append(origin + t * direction)
        verts = np.array(corners_3d, dtype=np.float32)
        norms = np.tile(view_dir.astype(np.float32), (4, 1))
        # Double-sided quad
        tris = np.array([
            [0, 1, 2], [0, 2, 3],
            [0, 2, 1], [0, 3, 2],
        ], dtype=np.int32)
        from chimerax.geometry import identity
        self._model.position = identity()
        self._model.set_geometry(verts, norms, tris)
        self._model.display = True

    def hide(self):
        if self._model is not None:
            self._model.display = False

    def delete(self):
        if self._model is not None:
            self._session.models.remove([self._model])
            self._model = None

class XRBackingWindow:
    '''
    Backing window for OpenXR autostereo 3D displays such as Acer SpatialLabs
    and Sony Spatial Reality to capture mouse and keyboard events when
    mouse is on the 3D display.
    '''
    def __init__(self, session, screen, in_front = False, hover_text = True,
                 direct_pick = False, cursor_3d = False):
        self._session = session
        self._screen = screen
        self._direct_pick = direct_pick
        self._cursor = None
        self._sel_rect = None
        self._sel_start = None

        # Create fullscreen backing Qt window on openxr screen.
        from Qt.QtWidgets import QWidget
        self._widget = w = QWidget()

        if in_front:
            self._make_transparent_in_front(w)

        w.move(screen.geometry().topLeft())
        w.showFullScreen()
        w.raise_()
        w.activateWindow()

        # Hover label state (own pause detection, independent of mouse_modes)
        self._hover_pos = None
        self._hover_time = 0
        self._hover_active = False

        # 3D cursor: hide OS cursor, render a 3D shape at scene depth.
        # Press C to cycle style: sphere → crosshair → diamond → arrow → pointer.
        self._cursor_styles = list(_CURSOR_STYLES) if cursor_3d else []
        self._cursor_style_index = 0
        if cursor_3d:
            from Qt.QtCore import Qt
            w.setCursor(Qt.BlankCursor)
            w.setMouseTracking(True)
            self._cursor = Cursor3D(session)

        self._register_mouse_handlers()

        # Forward key press events, intercepting cursor style toggle
        def key_press(event):
            from Qt.QtCore import Qt
            if self._cursor_styles and event.key() == Qt.Key_C and not event.modifiers():
                self._cycle_cursor_style()
            else:
                session.ui.forward_keystroke(event)
        w.keyPressEvent = key_press

        # Remove backing window when openxr is turned off.
        session.triggers.add_handler('vr stopped', self._xr_quit)

        # Show text labels for atoms and residues when mouse pauses.
        if hover_text:
            session.triggers.add_handler('graphics update',
                                         self._check_for_mouse_hover)

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

    def _cycle_cursor_style(self):
        '''Cycle 3D cursor style: sphere → crosshair → diamond → arrow → pointer.'''
        self._cursor_style_index = (self._cursor_style_index + 1) % len(self._cursor_styles)
        style = self._cursor_styles[self._cursor_style_index]
        if self._cursor is not None:
            self._cursor.set_style(style)
        self._session.logger.info(f'3D cursor: {style}')

    def _register_mouse_handlers(self):
        w = self._widget
        w.mousePressEvent = self._mouse_down
        w.mouseMoveEvent = self._mouse_drag
        w.mouseReleaseEvent = self._mouse_up
        w.mouseDoubleClickEvent = self._mouse_double_click
        w.wheelEvent = self._wheel

    def _mouse_down(self, event):
        # Track ctrl+left drag start for 3D selection rectangle
        from Qt.QtCore import Qt
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            p = event.position()
            if self._direct_pick:
                gx, gy = self._backing_to_render_coordinates(p.x(), p.y())
            else:
                gx, gy = self._backing_to_graphics_coordinates(p.x(), p.y())
            self._sel_start = (gx, gy)
            if self._sel_rect is None:
                self._sel_rect = SelectionRect3D(self._session)
        self._dispatch_mouse_event(event, "mouse_down")
    def _mouse_drag(self, event):
        # With mouse tracking enabled, mouseMoveEvent fires for all movement.
        # Only dispatch as drag when a button is actually pressed.
        if event.buttons():
            # Update 3D selection rectangle during ctrl+left drag
            from Qt.QtCore import Qt
            if self._sel_start is not None and event.buttons() & Qt.LeftButton:
                p = event.position()
                if self._direct_pick:
                    gx, gy = self._backing_to_render_coordinates(p.x(), p.y())
                else:
                    gx, gy = self._backing_to_graphics_coordinates(p.x(), p.y())
                self._sel_rect.update(
                    self._sel_start[0], self._sel_start[1], gx, gy)
            self._dispatch_mouse_event(event, "mouse_drag")
    def _mouse_up(self, event):
        if self._sel_start is not None:
            self._sel_start = None
            if self._sel_rect is not None:
                self._sel_rect.hide()
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
        if self._direct_pick:
            gx, gy = self._backing_to_render_coordinates(p.x(), p.y())
        else:
            gx, gy = self._backing_to_graphics_coordinates(p.x(), p.y())
        e = self._repositioned_event(event, gx, gy)
        mm = self._session.ui.mouse_modes
        mm._dispatch_mouse_event(e, action)

    def _dispatch_wheel_event(self, event):
        '''
        Convert a wheel event from 3D screen coordinates to
        graphics pane coordinates and dispatch it.
        '''
        p = event.position()
        gx, gy = self._backing_to_graphics_coordinates(p.x(), p.y())
        e = self._repositioned_event(event, gx, gy)
        mm = self._session.ui.mouse_modes
        mm._wheel_event(e)

    def _backing_to_graphics_coordinates(self, x, y):
        '''
        Convert backing window x,y pixel coordinates to main
        graphics window coordinates. Handle different aspect ratio
        of backing and graphics windows. Graphics window has cropped
        version of openxr window image.
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

    def _backing_to_render_coordinates(self, x, y):
        '''
        Map backing window coordinates directly to the XR per-eye
        render texture, bypassing the graphics pane aspect ratio
        correction. This is needed for vrto3d where the per-eye render
        (e.g. 1920x2160 portrait) has a very different aspect ratio from
        the graphics pane (e.g. 1979x1163 landscape).

        We compute what graphics pane coordinates would make ray()
        sample the correct position in the render texture by inverting
        the texture coordinate mapping that ray() applies.
        '''
        w3d = self._widget
        w, h = w3d.width(), w3d.height()
        if w == 0 or h == 0:
            return x, y
        cam = self._session.main_view.camera
        td = getattr(cam, '_texture_drawing', None)
        if td is None or td.texture is None:
            return self._backing_to_graphics_coordinates(x, y)
        fx, fy = x / w, y / h
        tc = td.texture_coordinates
        (xmin, ymin), (xmax, ymax) = tc[0], tc[2]
        gw, gh = self._session.main_view.window_size
        if (xmax - xmin) == 0 or (ymax - ymin) == 0:
            return self._backing_to_graphics_coordinates(x, y)
        gx = (fx - xmin) / (xmax - xmin) * gw
        gy = (fy - ymin) / (ymax - ymin) * gh
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

    def _graphics_cursor_position(self):
        from Qt.QtGui import QCursor
        cp = QCursor.pos()
        if self._session.ui.topLevelAt(cp) == self._widget:
            p = self._widget.mapFromGlobal(cp)
            x,y = self._backing_to_graphics_coordinates(p.x(), p.y())
            return (int(x), int(y))
        else:
            mm = self._session.ui.mouse_modes
            return mm._graphics_cursor_position_original()

    def _check_for_mouse_hover(self, *args):
        if self._widget is None:
            return 'delete handler'
        from Qt.QtGui import QCursor
        cp = QCursor.pos()
        if self._session.ui.topLevelAt(cp) != self._widget:
            if self._cursor is not None:
                self._cursor.hide()
            return

        bp = self._widget.mapFromGlobal(cp)
        if self._direct_pick:
            x, y = self._backing_to_render_coordinates(bp.x(), bp.y())
        else:
            x, y = self._backing_to_graphics_coordinates(bp.x(), bp.y())

        # Update 3D cursor position (once per frame, not per mouse event)
        if self._cursor is not None and self._cursor_styles:
            self._cursor.update(x, y)

        # Hover labels: detect pause ourselves (mouse_paused doesn't see
        # our backing window events, so we track position + time directly).
        import time
        ix, iy = int(x), int(y)
        now = time.time()
        if self._hover_pos != (ix, iy):
            # Mouse moved — reset timer, hide any active label
            self._hover_pos = (ix, iy)
            self._hover_time = now
            if self._hover_active:
                self._hover_active = False
                self._hide_hover_label()
        elif not self._hover_active and (now - self._hover_time) > 0.7:
            # Mouse paused for 0.7s — show label
            self._hover_active = True
            pick, object, label_type = self._hover_pick(x, y)
            if pick is None:
                self._hide_hover_label()
            else:
                self._show_hover_label(pick, object, label_type)

    def _show_hover_label(self, pick, object, label_type):
        text = pick.description()
        from chimerax.label.label3d import label
        label(self._session, object, label_type,
              text = text, bg_color = (0,0,0,255))
        self._hover_label_object = object, label_type

    def _hide_hover_label(self):
        if hasattr(self, '_hover_label_object'):
            object, label_type = self._hover_label_object
            if object:
                from chimerax.label.label3d import label_delete
                label_delete(self._session, object, label_type)
                self._hover_label_object = None, None

    def _hover_pick(self, x, y):
        pick = self._session.main_view.picked_object(x, y)

        from chimerax.atomic import PickedAtom, PickedResidue, PickedBond
        from chimerax.core.objects import Objects
        if isinstance(pick, PickedAtom):
            from chimerax.atomic import Atoms
            object = Objects(atoms = Atoms([pick.atom]))
            label_type = 'atoms'
        elif isinstance(pick, PickedResidue):
            object = Objects(atoms = pick.residue.atoms)
            label_type = 'residues'
        elif isinstance(pick, PickedBond):
            from chimerax.atomic import Bonds
            object = Objects(bonds = Bonds([pick.bond]))
            label_type = 'bonds'
        else:
            pick = object = label_type = None
        return pick, object, label_type

    def _xr_quit(self, *args):
        self._hide_hover_label()
        if self._cursor is not None:
            self._cursor.delete()
            self._cursor = None
        if self._sel_rect is not None:
            self._sel_rect.delete()
            self._sel_rect = None
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
