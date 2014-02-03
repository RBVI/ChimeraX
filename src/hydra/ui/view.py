from .qt import QtCore, QtGui, QtOpenGL, QtWidgets

class View(QtGui.QWindow):
    '''
    A View is the graphics windows that shows 3-dimensional models.
    It manages the camera and draws the models when needed.
    '''
    def __init__(self, session, parent=None):
        self.session = session
        QtGui.QWindow.__init__(self)
        self.widget = w = QtWidgets.QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QtGui.QSurface.OpenGLSurface)       # QWindow will be rendered with OpenGL
        w.setFocusPolicy(QtCore.Qt.ClickFocus)
# TODO: Qt 5.1 has touch events disabled on Mac
#        w.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        
        self.window_size = (w.width(), w.height())		# pixels
        self.background_rgba = (0,0,0,1)        # Red, green, blue, opacity, 0-1 range.

        # Determine stereo eye spacing parameter
        s = self.screen()
        eye_spacing = 61.0                      # millimeters
        ssize = s.physicalSize().width()        # millimeters
        psize = s.size().width()                # pixels
        eye_separation_pixels = psize * (eye_spacing / ssize)

        # Create camera
        from . import camera
        self.camera = camera.Camera(self.window_size, 'mono', eye_separation_pixels)
        '''The camera controlling the vantage shown in the graphics window.'''

        self.opengl_context = None

        from .. import draw
        self.render = draw.Render()

        self.timer = None			# Redraw timer
        self.redraw_interval = 16               # milliseconds
        self.redraw_needed = False
        self.block_redraw_count = 0
        self.new_frame_callbacks = []
        self.rendered_callbacks = []
        self.last_draw_duration = 0             # seconds

        self.overlays = []
        self.atoms_shown = 0

        from numpy import array, float32
        self.center_of_rotation = array((0,0,0), float32)
        self.update_center = True

        from . import mousemodes
        self.mouse_modes = mousemodes.Mouse_Modes(self)

    # QWindow method
    def resizeEvent(self, e):
        s = e.size()
        w, h = s.width(), s.height()
#
# TODO: On Mac retina display event window size is half of opengl window size.
#    Can scale width/height here, but also need mouse event positions to be scaled by 2x.
#    Not sure how to detect when app moves between non-retina and retina displays.
#    QWindow has a screenChanged signal but I did not get it in tests with Qt 5.2.
#    Also did not get moveEvent().  May need to get these on top level window?
#
#        r = self.devicePixelRatio()    # 2 on retina display, 1 on non-retina
#        w,h = int(r*w), int(r*h)
#
        self.window_size = w, h
        self.camera.window_size = w, h
        if not self.opengl_context is None:
            self.render.set_drawing_region(0,0,w,h)

    # QWindow method
    def exposeEvent(self, event):
        if self.isExposed():
            self.draw_graphics()

    def keyPressEvent(self, event):
        if str(event.text()) == '\r':
            return
        self.session.keyboard_shortcuts.key_pressed(event)

    def create_opengl_context(self):

        f = self.pixel_format(stereo = True)
        self.setFormat(f)
        self.create()

        c = QtGui.QOpenGLContext(self)
        c.setFormat(f)
        if not c.create():
            raise SystemError('Failed creating QOpenGLContext')
        c.makeCurrent(self)

        return c

    def pixel_format(self, stereo = False):

        f = QtGui.QSurfaceFormat()
        f.setMajorVersion(3)
        f.setMinorVersion(2)
        f.setDepthBufferSize(24);
        f.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        f.setStereo(stereo)
        return f

    def enable_opengl_stereo(self, enable):

        supported = self.opengl_context.format().stereo()
        if not enable or supported:
            return True

        msg = 'Stereo mode is not supported by OpenGL driver'
        s = self.session
        s.show_status(msg)
        s.show_info(msg)
        return False

        # TODO: Current strategy for handling stereo is to request a stereo OpenGL context
        # when graphics window created.  Use it for both stereo and mono display without
        # switching contexts. There are several obstacles to switching contexts.  First,
        # we need to share context state.  When tested with Qt 5.1 this caused crashes in
        # the QCocoaCreateOpenGLContext() routine, probably because the pixel format was null
        # perhaps because sharing was not supported.  A second problem is that we need to
        # switch the format of the QWindow.  It is not clear from the Qt documentation if this
        # is possible.  My tests failed.  The QWindow.setFormat() docs say "calling that function
        # after create() has been called will not re-resolve the surface format of the native surface."
        # Maybe calling destroy on the QWindow, then setFormat() and create() would work.  Did not try.
        # It may be necessary to simply destroy the old QWindow and QWidget container and make a new
        # one. A third difficulty is that OpenGL does not allow sharing VAOs between contexts.
        # Surface models use VAOs, so those would have to be destroyed and recreated.  Sharing does
        # handle VBOs, textures and shaders.
        #
        # Test code follows.
        #
        f = self.pixel_format(enable)
        c = QtGui.QOpenGLContext(self)
        c.setFormat(f)
        c.setShareContext(self.opengl_context)  # Share shaders, vbos and textures, but not VAOs.
        if not c.create() or (enable and not c.format().stereo()):
            if enable:
                msg = 'Stereo mode is not supported by OpenGL driver'
            else:
                msg = 'Failed changing graphics mode'
            s = self.session
            s.show_status(msg)
            s.show_info(msg)
            return False
        self.opengl_context = c
        c.makeCurrent(self)

        self.setFormat(f)
        if not self.create():
            raise SystemError('Failed to create QWindow with new format')

        return True

    def initialize_opengl(self):

        r = self.render
        r.set_background_color(self.background_rgba)
        r.enable_depth_test(True)
        r.initialize_opengl()

        s = self.session
        s.show_info('OpenGL version %s' % r.opengl_version())

        f = self.opengl_context.format()
        s.show_info('OpenGL stereo %d, color buffer size %d, depth buffer size %d, stencil buffer size %d'
                    % (f.stereo(), f.redBufferSize(), f.depthBufferSize(), f.stencilBufferSize()))

        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.initialize_llgr()

        if self.timer is None:
            self.start_update_timer()

    def use_opengl(self):
        if self.opengl_context is None:
            self.opengl_context = self.create_opengl_context()
            self.initialize_opengl()

        c = self.opengl_context
        c.makeCurrent(self)
        return c

    def draw_graphics(self):

        if not self.isExposed():
            return

        c = self.use_opengl()
        self.draw_scene()
        c.swapBuffers(self)

    def get_background_color(self):
        return self.background_rgba
    def set_background_color(self, rgba):
        self.background_rgba = tuple(rgba)
        self.redraw_needed = True
    background_color = property(get_background_color, set_background_color)

    def set_camera_mode(self, mode):
        '''
        Camera mode can be 'mono', 'stereo' for sequential stereo, or
        'oculus' for side-by-side parallel view stereo used by Oculus Rift goggles.
        '''
        c = self.camera
        if mode == c.mode:
            return True

        if mode == 'stereo' or c.mode == 'stereo':
            if not self.enable_opengl_stereo(mode == 'stereo'):
                return False
        elif not mode in ('mono', 'oculus'):
            raise ValueError('Unknown camera mode %s' % mode)

        c.mode = mode
        self.redraw_needed = True

    def add_overlay(self, overlay):
        self.overlays.append(overlay)
        self.redraw_needed = True

    def remove_overlays(self, models = None):
        if models is None:
            models = self.overlays
        for o in models:
            o.delete()
        oset = set(models)
        self.overlays = [o for o in self.overlays if not o in oset]
        self.redraw_needed = True

    def image(self, width = None, height = None, camera = None, models = None):

        self.use_opengl()

        w = self.window_size[0] if width is None else width
        h = self.window_size[1] if height is None else height

        r = self.render
        if not r.render_to_buffer(w,h):
            return None

        # Camera needs correct aspect ratio when setting projection matrix.
        c = camera if camera else self.camera
        prev_size = c.window_size
        c.window_size = (width,height)

        self.draw_scene(c, models)

        c.window_size = prev_size

        rgb = r.frame_buffer_image(w, h, r.IMAGE_FORMAT_RGB32)
        r.render_to_screen()
        ww, wh = self.window_size
        r.set_drawing_region(0,0,ww,wh)
        qi = QtGui.QImage(rgb, w, h, QtGui.QImage.Format_RGB32)
        return qi

    def start_update_timer(self):

        self.timer = t = QtCore.QTimer(self)
        t.timeout.connect(self.redraw)
        t.start(self.redraw_interval)

    def renderer(self):
        return self.render

    def redraw(self):

        if self.block_redraw_count == 0:
            # Avoid redrawing during callbacks of the current redraw.
            self.block_redraw()
            try:
                self.redraw_graphics()
            finally:
                self.unblock_redraw()

    def redraw_graphics(self):
        for cb in self.new_frame_callbacks:
            cb()

        c = self.camera
        s = self.session
        draw = self.redraw_needed or c.redraw_needed or s.redraw_needed
        mlist = s.model_list() + self.overlays
        if draw:
            for m in mlist:
                m.redraw_needed = False
        else:
            for m in mlist:
                if m.redraw_needed:
                    m.redraw_needed = False
                    draw = True
        if draw:
            self.redraw_needed = False
            c.redraw_needed = False
            s.redraw_needed = False
            self.draw_graphics()
            for cb in self.rendered_callbacks:
                cb()
        else:
            self.mouse_modes.mouse_pause_tracking()

    def block_redraw(self):
        self.block_redraw_count += 1
    def unblock_redraw(self):
        self.block_redraw_count -= 1

    def add_new_frame_callback(self, cb):
        '''Add a function to be called before each redraw.  The function takes no arguments.'''
        self.new_frame_callbacks.append(cb)
    def remove_new_frame_callback(self, cb):
        '''Add a callback that was added with add_new_frame_callback().'''
        self.new_frame_callbacks.remove(cb)

    def add_rendered_frame_callback(self, cb):
        '''Add a function to be called after each redraw.  The function takes no arguments.'''
        self.rendered_callbacks.append(cb)
    def remove_rendered_frame_callback(self, cb):
        '''Add a callback that was added with add_rendered_frame_callback().'''
        self.rendered_callbacks.remove(cb)

    def draw_scene(self, camera = None, models = None):
        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.render(self)
            return

        if camera is None:
            camera = self.camera
        if models is None:
            models = [m for m in self.session.model_list() if m.display]

        r = self.render
        r.set_background_color(self.background_rgba)

        self.update_level_of_detail()

        from time import process_time
        t0 = process_time()
        for vnum in range(camera.number_of_views()):
            camera.setup(vnum, r)
            if models:
                self.draw(self.OPAQUE_DRAW_PASS, vnum, camera, models)
                if self.session.transparent_models_shown():
                    r.draw_transparent(lambda: self.draw(self.TRANSPARENT_DEPTH_DRAW_PASS, vnum, camera, models),
                                       lambda: self.draw(self.TRANSPARENT_DRAW_PASS, vnum, camera, models))
            s = camera.finish_draw(vnum, r)
            if s:
                self.draw_overlays([s])
        t1 = process_time()
        self.last_draw_duration = t1-t0

        if self.overlays:
            self.draw_overlays(self.overlays)

    def draw_overlays(self, overlays):

        i = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        r = self.render
        r.set_projection_matrix(i)
        r.set_model_view_matrix(matrix = i)
        r.enable_depth_test(False)
        for m in overlays:
            m.draw(self, self.OPAQUE_DRAW_PASS)
        r.enable_blending(True)
        for m in overlays:
            m.draw(self, self.TRANSPARENT_DRAW_PASS)
        r.enable_depth_test(True)

    OPAQUE_DRAW_PASS = 'opaque'
    TRANSPARENT_DRAW_PASS = 'transparent'
    TRANSPARENT_DEPTH_DRAW_PASS = 'transparent depth'

    def draw(self, draw_pass, view_num, camera, models):

        self.update_projection(view_num, camera = camera)
        for m in models:
            self.draw_model(m, draw_pass, view_num, camera)

    def draw_model(self, m, draw_pass, view_num, camera):
        cvinv = camera.view_inverse(view_num)
        r = self.render
        if m.copies:
            for p in m.copies:
                r.set_model_view_matrix(cvinv, p)
                m.draw(self, draw_pass)
        else:
            r.set_model_view_matrix(cvinv, m.place)
            m.draw(self, draw_pass)

    def update_level_of_detail(self):
        # Level of detail updating.
        # TODO: Don't recompute atoms shown on every draw, only when changed
        ashow = sum(m.shown_atom_count() for m in self.session.molecules() if m.display)
        if ashow != self.atoms_shown:
            self.atoms_shown = ashow
            for m in self.session.molecules():
                m.update_level_of_detail(self)

    def initial_camera_view(self):

        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        from numpy import array, float32
        self.camera.initialize_view(center, s)
        self.center_of_rotation = center

    def view_all(self):
        '''Adjust the camera to show all displayed models.'''
        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        shift = self.camera.view_all(center, s)
        self.translate(-shift, update_clip_planes = False)

    def center_of_rotation_needs_update(self):
        self.update_center = True

    def update_center_of_rotation(self):
        if not self.update_center:
            return
        self.update_center = False
        center, s = self.session.bounds_center_and_width()
        if center is None:
            return
        vw = self.camera.view_width(center)
        if vw >= s:
            # Use center of models for zoomed out views
            cr = center
        else:
            # Use front center point for zoomed in views
            cr = self.front_center_point()
            if cr is None:
                return
        self.center_of_rotation = cr
        self.camera.set_near_far_clip(center, s)

    def front_center_point(self):
        w, h = self.window_size
        p, s = self.first_intercept(0.5*w, 0.5*h)
        return p

    def first_intercept(self, win_x, win_y):
        xyz1, xyz2 = self.camera.clip_plane_points(win_x, win_y)
        f = None
        s = None
        models = self.session.model_list()
        for m in models:
            if m.display:
                mxyz1, mxyz2 = m.place.inverse() * (xyz1,xyz2)
                fmin, smin = m.first_intercept(mxyz1, mxyz2)
                if not fmin is None and (f is None or fmin < f):
                    f = fmin
                    s = smin
        if f is None:
            return None, None
        p = (1.0-f)*xyz1 + f*xyz2
        return p, s

    def update_projection(self, view_num = None, win_size = None, camera = None):
        
        c = self.camera if camera is None else camera
        ww,wh = c.window_size if win_size is None else win_size
        if ww > 0 and wh > 0:
            pm = c.projection_matrix(view_num, (ww,wh))
            self.render.set_projection_matrix(pm)

    def rotate(self, axis, angle, models = None):
        '''
        Move camera to simulate a rotation of models about current rotation center.
        Axis is in scene coordinates and angle is in degrees.
        '''
        self.update_center_of_rotation()
        from ..geometry import place
        r = place.rotation(axis, angle, self.center_of_rotation)
        self.move(r, models)

    def translate(self, shift, models = None, update_clip_planes = True):
        '''Move camera to simulate a translation of models.  Translation is in scene coordinates.'''
        self.center_of_rotation_needs_update()
        from ..geometry import place
        t = place.translation(shift)
        self.move(t, models, update_clip_planes)

    def move(self, tf, models = None, update_clip_planes = False):
        '''Move camera to simulate a motion of models.'''
        if models is None:
            c = self.camera
            cv = c.view()
            c.set_view(tf.inverse() * cv)
        else:
            for m in models:
                m.place = tf * m.place
        if update_clip_planes:
            cr = self.center_of_rotation
            shift = (tf*cr) - cr
            c = self.camera
            dz = c.view_inverse().apply_without_translation(shift)[2]
            c.shift_near_far_clip(-dz)

        self.redraw_needed = True

    def pixel_size(self, p = None):
        '''Return the pixel size in scene length units at point p in the scene.'''
        return self.camera.pixel_size(self.center_of_rotation if p is None else p)
