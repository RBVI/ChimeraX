from .qt import QtCore, QtGui, QtOpenGL, QtWidgets

class View(QtGui.QWindow):
    '''
    A View is the graphics windows that shows 3-dimensional models.
    It manages the camera and draws the models when needed.
    Currently it contains the list of open models.
    '''
    def __init__(self, parent=None):
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

        self.tile = False
        self.tile_edge_color = (.3,.3,.3,1)
        self.tile_scale = 0
        self.tile_animation_steps = 10

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

        self.models = []
        self.next_id = 1
        self.overlays = []
        self.selected = set()
        self.atoms_shown = 0

        self.center_of_rotation = (0,0,0)
        self.update_center = True

        from . import mousemodes
        self.mouse_modes = mousemodes.Mouse_Modes(self)

    # QWindow method
    def resizeEvent(self, e):
        s = e.size()
        w, h = s.width(), s.height()
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
        from .shortcuts import keyboard_shortcuts as ks
        ks.key_pressed(event)

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
        from .gui import show_status, show_info
        show_status(msg)
        show_info(msg)
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
            from .gui import show_status, show_info
            show_status(msg)
            show_info(msg)
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

        from .gui import show_info
        show_info('OpenGL version %s' % r.opengl_version())

        f = self.opengl_context.format()
        show_info('OpenGL stereo %d, color buffer size %d, depth buffer size %d, stencil buffer size %d'
                  % (f.stereo(), f.redBufferSize(), f.depthBufferSize(), f.stencilBufferSize()))

        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.initialize_llgr()

        if self.timer is None:
            self.start_update_timer()

    def draw_graphics(self):

        if not self.isExposed():
            return
        if self.opengl_context is None:
            self.opengl_context = self.create_opengl_context()
            self.initialize_opengl()

        c = self.opengl_context
        c.makeCurrent(self)
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

    def add_model(self, model):
        '''
        Add a model to the scene.  A model is a Surface object.
        '''
        self.models.append(model)
        if model.id is None:
            model.id = self.next_id
            self.next_id += 1
        from ..map.volume import Volume, volume_manager
        if isinstance(model, Volume):
            if not model in volume_manager.data_regions:
                volume_manager.add_volume(model)
        if model.display:
            self.redraw_needed = True

    def add_models(self, mlist):
        '''
        Add a list of models to the scene.
        '''
        for m in mlist:
            self.add_model(m)
        
    def close_models(self, mlist):
        '''
        Remove a list of models from the scene.
        '''
        from ..map.volume import volume_manager, Volume
        vlist = [m for m in mlist if isinstance(m, Volume)]
        volume_manager.remove_volumes(vlist)
        olist = self.models
        for m in mlist:
            olist.remove(m)
            self.selected.discard(m)
            if m.display:
                self.redraw_needed = True
            m.delete()
        self.next_id = 1 if len(olist) == 0 else max(m.id for m in olist) + 1
        
    def close_all_models(self):
        '''
        Remove all models from the scene.
        '''
        self.close_models(tuple(self.models))

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

    def image(self, size = None):
        w,h = self.window_size
        r = self.render
        rgb = r.frame_buffer_image(w, h, r.IMAGE_FORMAT_RGB32)
        qi = QtGui.QImage(rgb, w, h, QtGui.QImage.Format_RGB32)
        if not size is None:
            sw,sh = size
            if sw*h < sh*w:
                sh = max(1,(h*sw)/w)
            elif sw*h > sh*w:
                sw = max(1,(w*sh)/h)
            qi = qi.scaled(sw, sh, QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)
        return qi

    def select_model(self, m):
        self.selected.add(m)
        if m.display:
            self.redraw_needed = True

    def unselect_model(self, m):
        self.selected.discard(m)
        if m.display:
            self.redraw_needed = True

    def clear_selection(self):
        for m in self.selected:
            m.selected = False
        if self.selected:
            self.redraw_needed = True
        self.selected.clear()

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
        draw = self.redraw_needed or c.redraw_needed
        if draw:
            for m in self.models + self.overlays:
                m.redraw_needed = False
        else:
            for m in self.models + self.overlays:
                if m.redraw_needed:
                    m.redraw_needed = False
                    draw = True
        if draw:
            self.redraw_needed = False
            c.redraw_needed = False
            self.draw_graphics()
            for cb in self.rendered_callbacks:
                cb()
        else:
            self.mouse_modes.mouse_pause_tracking()

    def block_redraw(self):
        self.block_redraw_count += 1
    def unblock_redraw(self):
        self.block_redraw_count -= 1

    def transparent_models_shown(self):

        for m in self.models:
            if m.display and m.showing_transparent():
                return True
        return False

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

    def draw_scene(self):
        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.render(self)
            return

        r = self.render
        r.set_background_color(self.background_rgba)

        if self.models:
            self.update_level_of_detail()

            from time import process_time
            t0 = process_time()
            c = self.camera
            for vnum in range(c.number_of_views()):
                c.setup(vnum, r)
                self.draw(self.OPAQUE_DRAW_PASS, vnum)
                if self.transparent_models_shown():
                    r.draw_transparent(lambda: self.draw(self.TRANSPARENT_DEPTH_DRAW_PASS, vnum),
                                       lambda: self.draw(self.TRANSPARENT_DRAW_PASS, vnum))
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

    def draw(self, draw_pass, view_num):

        models = self.models
        n = len(models)
        draw_tiles = (self.tile_scale > 0)
        if draw_tiles:
            r = self.render
            tiles = self.tiles(self.tile_scale)
            if draw_pass == self.OPAQUE_DRAW_PASS:
                self.next_tile_size()
                if self.tile_scale >= 1:
                    fill = [m.display for m in models]
                    r.draw_tile_outlines(tiles, self.tile_edge_color,
                                         self.background_rgba, fill)
            x,y,w,h = tiles[0]
            r.set_drawing_region(x,y,w,h)
            self.update_projection(view_num, (w,h))
        else:
            self.update_projection(view_num)

        for m in models:
            if m.display:
                self.draw_model(m, draw_pass, view_num)

        if draw_tiles:
            if n > 1:
                for i,m in enumerate(models):
                    x,y,w,h = tiles[i+1]
                    r.set_drawing_region(x,y,w,h)
                    self.update_projection(view_num, (w,h))
                    self.draw_model(m, draw_pass, view_num)
                    if self.tile_scale >= 1:
                        self.draw_caption('#%d %s' % (m.id, m.name))
                w,h = self.window_size
                r.set_drawing_region(0,0,w,h)
            elif n == 1:
                m = self.models[0]
                self.draw_caption('#%d %s' % (m.id, m.name))

    def draw_model(self, m, draw_pass, view_num):
        cvinv = self.camera.view_inverse(view_num)
        r = self.render
        if m.copies:
            for p in m.copies:
                r.set_model_view_matrix(cvinv, p)
                m.draw(self, draw_pass)
        else:
            r.set_model_view_matrix(cvinv, m.place)
            m.draw(self, draw_pass)

    def draw_caption(self, text):

        # TODO: reuse image and surface and texture for each caption.
        width, height = 512,64
        im = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        im.fill(QtGui.QColor(*tuple(int(255*c) for c in self.background_color)))
        from . import qt
        qt.draw_image_text(im, text, bgcolor = self.background_color)
        from ..surface import Surface, surface_image
        surf = Surface('Caption')
        pos = -.95,-1     # x,y range -1 to 1
        size = 1.9,.25
        surface_image(im, pos, size, surf)
        self.draw_overlays([surf])
        surf.delete()

    def get_tile_models(self):
        return self.tile
    def set_tile_models(self, tile):
        self.tile = tile
        if len(self.models) <= 1:
            ts = 1 if tile else 0
        elif self.last_draw_duration < .05:
            step = 1.0/self.tile_animation_steps
            ts = step if tile else 1-step
        else:
            ts = 1 if tile else 0
        self.tile_scale = ts
        self.redraw_needed = True
    tile_models = property(get_tile_models, set_tile_models)

    def next_tile_size(self):
        ts = self.tile_scale
        steps = self.tile_animation_steps
        if self.tile:
            if ts < 1:
                self.tile_scale = min(1, ts+1.0/steps)
                self.redraw_needed = True
        else:
            if ts > 0:
                self.tile_scale = max(0, ts-1.0/steps)
                self.redraw_needed = True

    def tiles(self, scale = 1):

        w,h = self.window_size
        n = len(self.models)
        if n == 1:
            tiles = [(0, 0, w, h)]
        elif n <= 4:
            if h >= w:
                # Single row of thumbnails along bottom
                ts = w//n
                sts = max(1,int(scale*ts))
                tiles = [(0, sts, w, h-sts)]
                for t in range(n):
                    tiles.append((t*ts, 0, sts, sts))
            else:
                # Single column of thumbnails along right edge
                ts = h//n
                sts = max(1,int(scale*ts))
                tiles = [(0, 0, w-sts, h)]
                for t in range(n):
                    tiles.append((w-sts, h-(t+1)*ts, sts, sts))
        elif False:
            # Thumbnails along bottom and right edges.
            div = max(3,(n//2)+1)
            wd, hd = w//div, h//div
            swd, shd = [max(1,int(scale*ts)) for ts in (wd,hd)]
            tiles = [(0, shd, w-swd, h-shd)]
            for d in range(div):
                tiles.append((d*wd,0,swd,shd))
            for d in range(1,div):
                tiles.append((w-swd,d*hd,swd,shd))
        else:
            tfrac = 0.75
            from math import sqrt
            ts = max(1, int(sqrt(tfrac*w*h/(n+1))))
            while True:
                cols = w//ts
                rows = h//ts
                if rows*cols > n or ts == 1:
                    break
                ts -= 1
            ts = min(w//cols, h//rows)
            n0 = int(sqrt(rows*cols - n))
            tiles = [(0,0,n0*ts,n0*ts)]
            for r in range(n0):
                for c in range(n0,cols):
                    tiles.append((c*ts,r*ts,ts,ts))
            for r in range(n0,rows):
                for c in range(cols):
                    tiles.append((c*ts,r*ts,ts,ts))
            tiles = tiles[:n+1]
            tiles = [(x,h-th-y,tw,th) for x,y,tw,th in tiles]   # Flip vertically
            if scale < 1:
                stiles = []
                x0,y0,tw,th = tiles[0]
                s = scale
                stiles.append((x0,int(s*y0),int(s*tw + (1-s)*w),int(s*th + (1-s)*h)))
                stiles.extend([(x,int(y*s),int(tw*s),int(th*s)) for x,y,tw,th in tiles[1:]])
                tiles = stiles

        return tiles

    def update_level_of_detail(self):
        # Level of detail updating.
        # TODO: Don't recompute atoms shown on every draw, only when changed
        ashow = sum(m.shown_atom_count() for m in self.molecules() if m.display)
        if ashow != self.atoms_shown:
            self.atoms_shown = ashow
            for m in self.molecules():
                m.update_level_of_detail(self)

    def initial_camera_view(self):

        center, s = self.bounds_center_and_width()
        if center is None:
            return
        self.camera.initialize_view(center, s)
        self.center_of_rotation = center

    def view_all(self):
        '''Adjust the camera to show all displayed models.'''
        center, s = self.bounds_center_and_width()
        if center is None:
            return
        shift = self.camera.view_all(center, s)
        csx,csy,csz = self.camera.view_inverse().apply_without_translation(shift)
        self.translate(-csx,-csy,-csz)

    def center_of_rotation_needs_update(self):
        self.update_center = True

    def update_center_of_rotation(self):
        if not self.update_center:
            return
        self.update_center = False
        center, s = self.bounds_center_and_width()
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
        self.center_of_rotation = tuple(cr)
        self.camera.set_near_far_clip(center, s)

    def front_center_point(self):
        w, h = self.window_size
        p, s = self.first_intercept(0.5*w, 0.5*h)
        return p

    def first_intercept(self, win_x, win_y):
        xyz1, xyz2 = self.camera.clip_plane_points(win_x, win_y)
        f = None
        s = None
        for m in self.models:
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

    def bounds_center_and_width(self):

        bounds = self.bounds()
        if bounds is None or bounds == (None, None):
            return None, None
        (xmin,ymin,zmin), (xmax,ymax,zmax) = bounds
        w = max(xmax-xmin, ymax-ymin, zmax-zmin)
        cx,cy,cz = 0.5*(xmin+xmax),0.5*(ymin+ymax),0.5*(zmin+zmax)
        from numpy import array
        return array((cx,cy,cz)), w

    def bounds(self):
        from ..geometry import bounds
        b = bounds.union_bounds(m.placed_bounds() for m in self.models if m.display)
        return b

    def update_projection(self, view_num = None, win_size = None):
        
        ww,wh = self.window_size if win_size is None else win_size
        if ww > 0 and wh > 0:
            pm = self.camera.projection_matrix(view_num, (ww,wh))
            self.render.set_projection_matrix(pm)

    def rotate(self, axis, angle, models = None):

        self.update_center_of_rotation()
        # Rotation axis is in camera coordinates.
        # Center of rotation is in model coordinates.
        c = self.camera
        cv = c.view()
        from ..geometry import place
        maxis = cv.apply_without_translation(axis)
        r = place.rotation(maxis, angle, self.center_of_rotation)
        if models is None:
            c.set_view(r.inverse() * cv)
        else:
            for m in models:
                m.place = r * m.place
        self.redraw_needed = True

    # Translation is in camera coordinates.  Sign is for moving models.
    def translate(self, dx, dy, dz, models = None):

        self.center_of_rotation_needs_update()
        c = self.camera
        cv = c.view()
        from ..geometry import place
        mt = cv.apply_without_translation((dx,dy,dz))
        t = place.translation(mt)
        if models is None:
            c.set_view(t.inverse()*cv)
            c.shift_near_far_clip(-dz)
        else:
            for m in models:
                m.place = t * m.place
        self.redraw_needed = True

    def pixel_size(self, p = None):
        '''Return the pixel size in scene length units at point p in the scene.'''
        return self.camera.pixel_size(self.center_of_rotation if p is None else p)

    def maps(self):
        '''Return a list of the Volume models in the scene.'''
        from ..map import Volume
        return tuple(m for m in self.models if isinstance(m,Volume))

    def molecules(self):
        '''Return a list of the Molecule models in the scene.'''
        from ..molecule import Molecule
        return tuple(m for m in self.models if isinstance(m,Molecule))

    def surfaces(self):
        '''Return a list of the Surface models in the scene which are not Molecules.'''
        from ..molecule import Molecule
        return tuple(m for m in self.models if not isinstance(m,(Molecule)))
