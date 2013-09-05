from .qt import QtCore, QtGui, QtOpenGL

class View(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
#        self.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
#        self.setAutoBufferSwap(False)
        
        # Camera postion and direction, neg z-axis is camera view direction,
        # x and y axes are horizontal and vertical screen axes.
        # First 3 columns are x,y,z axes, 4th column is camara location.
        from ..geometry.place import Place
        self.camera_view = self.camera_view_inverse = Place()
        self.field_of_view = 45   	# degrees, width
        self.center_of_rotation = (0,0,0)
        self.update_center = True
        self.near_far_clip = (1,100)      # along -z in camera coordinates
        self.window_size = (800,800)	# pixels
#        self.window_size = (1024,1024)
        self.background_rgba = (0,0,0,1)

        self.tile = False
        self.tile_edge_color = (.3,.3,.3,1)
        self.tile_scale = 0
        self.tile_animation_steps = 10
        
        # Lighting parameters
        self.key_light_position = (-.577,.577,.577)
        self.key_light_diffuse_color = (.6,.6,.6)
        self.key_light_specular_color = (.3,.3,.3)
        self.key_light_specular_exponent = 20
        self.fill_light_position = (.2,.2,.959)
        self.fill_light_diffuse_color = (.3,.3,.3)
        self.ambient_light_color = (.3,.3,.3)

        from ..draw import drawing
        self.render = drawing.Renderer(self)

        self.timer = None			# Redraw timer
        self.redraw_interval = 16               # milliseconds
        self.redraw_needed = False
        self.new_frame_callbacks = []
        self.rendered_callbacks = []
        self.last_draw_duration = 0             # seconds

        self.models = []
        self.next_id = 1
        self.overlays = []
        self.selected = set()
        self.atoms_shown = 0

        self.mouse_modes = {}
        self.last_mouse_position = None
        self.last_mouse_time = None
        self.mouse_pause_interval = 0.5         # seconds
        self.mouse_pause_position = None
        self.mouse_perimeter = False
        self.wheel_function = None
        self.bind_standard_mouse_modes()

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        w,h = self.window_size
        return QtCore.QSize(w, h)

    def get_background_color(self):
        return self.background_rgba
    def set_background_color(self, rgba):
        self.background_rgba = tuple(rgba)
        self.redraw_needed = True
    background_color = property(get_background_color, set_background_color)

    def add_model(self, model):
        self.models.append(model)
        model.id = self.next_id
        self.next_id += 1
        from ..VolumeViewer.volume import Volume, volume_manager
        if isinstance(model, Volume):
            if not model in volume_manager.data_regions:
                volume_manager.add_volume(model)
        if model.display:
            self.redraw_needed = True

    def add_models(self, mlist):
        for m in mlist:
            self.add_model(m)
        
    def close_models(self, mlist):
        from ..VolumeViewer.volume import volume_manager, Volume
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
        from ..draw import drawing
        rgb = drawing.frame_buffer_image(w, h)
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

    def initializeGL(self):

        from ..draw import drawing
        drawing.set_background_color(self.background_rgba)
        drawing.enable_depth_test(True)
        drawing.initialize_opengl()

        from .gui import show_info
        show_info('OpenGL version %s' % drawing.opengl_version())

        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.initialize_llgr()

        self.start_update_timer()

    def start_update_timer(self):

        self.timer = t = QtCore.QTimer(self)
        t.timeout.connect(self.redraw_graphics)
        t.start(self.redraw_interval)

    def set_shader(self, **kw):
        return self.render.use_shader(**kw)

    def current_shader(self):
        return self.render.current_shader_program

    def redraw_graphics(self):

        for cb in self.new_frame_callbacks:
            cb()

        draw = self.redraw_needed
        if draw:
            for m in self.models:
                m.redraw_needed = False
        else:
            for m in self.models:
                if m.redraw_needed:
                    m.redraw_needed = False
                    draw = True
        if draw:
            self.redraw_needed = False
            self.updateGL()
            for cb in self.rendered_callbacks:
                cb()
        else:
            self.mouse_pause_tracking()

    def transparent_models_shown(self):

        for m in self.models:
            if m.display and m.showing_transparent():
                return True
        return False

    def add_new_frame_callback(self, cb):
        self.new_frame_callbacks.append(cb)
    def remove_new_frame_callback(self, cb):
        self.new_frame_callbacks.remove(cb)

    def add_rendered_frame_callback(self, cb):
        self.rendered_callbacks.append(cb)
    def remove_rendered_frame_callback(self, cb):
        self.rendered_callbacks.remove(cb)

    def paintGL(self):
        from ..draw import llgrutil as gr
        if gr.use_llgr:
            gr.render(self)
            return

        from ..draw import drawing
        drawing.set_background_color(self.background_rgba)
        drawing.draw_background()

        if self.models:
            self.update_level_of_detail()

            from time import process_time
            t0 = process_time()
            self.draw(self.OPAQUE_DRAW_PASS)
            if self.transparent_models_shown():
                drawing.draw_transparent(lambda: self.draw(self.TRANSPARENT_DEPTH_DRAW_PASS),
                                         lambda: self.draw(self.TRANSPARENT_DRAW_PASS))
            t1 = process_time()
            self.last_draw_duration = t1-t0

        if self.overlays:
            self.draw_overlays(self.overlays)

    def draw_overlays(self, overlays):

        p = self.current_shader()
        if p is None:
            return

        i = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        self.render.set_projection_matrix(i)
        self.render.set_model_view_matrix(matrix = i)
        from ..draw import drawing
        drawing.enable_depth_test(False)
        for m in overlays:
            m.draw(self, self.OPAQUE_DRAW_PASS)
            m.draw(self, self.TRANSPARENT_DRAW_PASS)
        drawing.enable_depth_test(True)

    OPAQUE_DRAW_PASS = 'opaque'
    TRANSPARENT_DRAW_PASS = 'transparent'
    TRANSPARENT_DEPTH_DRAW_PASS = 'transparent depth'

    def draw(self, draw_pass):

        models = self.models
        n = len(models)
        draw_tiles = (self.tile_scale > 0)
        if draw_tiles:
            from ..draw.drawing import set_drawing_region, draw_tile_outlines
            tiles = self.tiles(self.tile_scale)
            if draw_pass == self.OPAQUE_DRAW_PASS:
                self.next_tile_size()
                if self.tile_scale >= 1:
                    fill = [m.display for m in models]
                    draw_tile_outlines(tiles, self.tile_edge_color,
                                       self.background_rgba, fill)
            x,y,w,h = tiles[0]
            set_drawing_region(x,y,w,h)
            self.update_projection((w,h))
        else:
            self.update_projection()

        for m in models:
            if m.display:
                self.draw_model(m, draw_pass)

        if draw_tiles:
            if n > 1:
                for i,m in enumerate(models):
                    x,y,w,h = tiles[i+1]
                    set_drawing_region(x,y,w,h)
                    self.update_projection((w,h))
                    self.draw_model(m, draw_pass)
                    if self.tile_scale >= 1:
                        self.draw_caption('#%d %s' % (m.id, m.name))
                w,h = self.window_size
                set_drawing_region(0,0,w,h)
            elif n == 1:
                m = self.models[0]
                self.draw_caption('#%d %s' % (m.id, m.name))

    def draw_model(self, m, draw_pass):
        if m.copies:
            for p in m.copies:
                self.render.set_model_view_matrix(self.camera_view_inverse, p)
                m.draw(self, draw_pass)
        else:
            self.render.set_model_view_matrix(self.camera_view_inverse, m.place)
            m.draw(self, draw_pass)

    def draw_caption(self, text):

        # TODO: reuse image and surface and texture for each caption.
        width, height = 256,32
        im = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        im.fill(QtGui.QColor(*tuple(int(255*c) for c in self.background_color)))
        draw_image_text(im, text, bgcolor = self.background_color)
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
            ts = int(sqrt(tfrac*w*h/(n+1)))
            cols = w//ts
            rows = h//ts
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
        ashow = sum(m.atoms_shown() for m in self.molecules() if m.display)
        if ashow != self.atoms_shown:
            self.atoms_shown = ashow
            for m in self.molecules():
                m.update_level_of_detail(self)

    def resizeGL(self, width, height):
        self.window_size = width, height
        from ..draw import drawing
        drawing.set_drawing_region(0,0,width,height)

    def initial_camera_view(self):

        center, s = self.bounds_center_and_width()
        if center is None:
            return
        cx,cy,cz = center
        from math import pi, tan
        fov = self.field_of_view*pi/180
        camdist = 0.5*s + 0.5*s/tan(0.5*fov)
        from ..geometry import place
        self.set_camera_view(place.translation((cx,cy,cz+camdist)))
        self.near_far_clip = (camdist - s, camdist + s)
        self.center_of_rotation = (cx,cy,cz)

    def view_all(self):

        center, s = self.bounds_center_and_width()
        if center is None:
            return
        from math import pi, tan
        fov = self.field_of_view*pi/180
        d = 0.5*s + 0.5*s/tan(0.5*fov)
        vd = self.view_direction()
        cp = self.camera_position()
        shift = tuple((center[a]-d*vd[a])-cp[a] for a in (0,1,2))
        cv = self.camera_view
        csx,csy,csz = cv.inverse().apply_without_translation(shift)
        self.translate(-csx,-csy,-csz)
        self.near_far_clip = (d - s, d + s)

    def set_camera_view(self, place):
        self.camera_view = place
        self.camera_view_inverse = place.inverse()
        self.redraw_needed = True

    def center_of_rotation_needs_update(self):
        self.update_center = True

    def update_center_of_rotation(self):
        if not self.update_center:
            return
        self.update_center = False
        center, s = self.bounds_center_and_width()
        if center is None:
            return
        cp = self.camera_position()
        vd = self.view_direction()
        d = sum((center-cp)*vd)         # camera to center of models
        from math import tan, pi
        vw = 2*d*tan(0.5*self.field_of_view*pi/180)     # view width at center of models
        if vw >= s:
            # Use center of models for zoomed out views
            cr = center
        else:
            # Use front center point for zoomed in views
            cr = self.front_center_point()
            if cr is None:
                return
        self.center_of_rotation = tuple(cr)
        self.near_far_clip = (d - s, d + s)

    def camera_position(self):
        return self.camera_view.translation()

    def view_direction(self):
        return -self.camera_view.z_axis()

    def front_center_point(self):
        w, h = self.window_size
        p, s = self.first_intercept(0.5*w, 0.5*h)
        return p

    def first_intercept(self, win_x, win_y):
        xyz1, xyz2 = self.clip_plane_points(win_x, win_y)
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
        from .. import surface
        b = surface.union_bounds(m.placed_bounds() for m in self.models if m.display)
        return b

    def update_projection(self, win_size = None):
        
        ww,wh = self.window_size if win_size is None else win_size
        if ww > 0 and wh > 0:
            pm = self.projection_matrix((ww,wh))
            self.render.set_projection_matrix(pm)

    def projection_matrix(self, win_size = None):

        # Perspective projection to origin with center of view along z axis
        from math import pi, tan
        fov = self.field_of_view*pi/180
        near,far = self.near_far_clip
        near = max(near, 1)
        far = max(far, near+1)
        w = 2*near*tan(0.5*fov)
        ww,wh = self.window_size if win_size is None else win_size
        aspect = float(wh)/ww
        h = w*aspect
        left, right, bot, top = -0.5*w, 0.5*w, -0.5*h, 0.5*h
        pm = frustum(left, right, bot, top, near, far)
        return pm

    def model_view_matrix(self, model):

        mv44 = (self.camera_view_inverse * model.place).opengl_matrix()
        return mv44

    # Returns camera coordinates.
    def camera_clip_plane_points(self, window_x, window_y):
        znear, zfar = self.near_far_clip
        from math import pi, tan
        fov = self.field_of_view*pi/180
        wn = 2*znear*tan(0.5*fov)   # Screen width in model units, near clip
        wf = 2*zfar*tan(0.5*fov)    # Screen width in model units, far clip
        wp,hp = self.window_size     # Screen size in pixels
        rn, rf = (wn/wp, wf/wp) if wp != 0 else (1,1)
        wx,wy = window_x - 0.5*wp, -(window_y - 0.5*hp)
        cn = (rn*wx, rn*wy, -znear)
        cf = (rf*wx, rf*wy, -zfar)
        return cn, cf

    # Returns scene coordinates.
    def clip_plane_points(self, window_x, window_y):
        cn, cf = self.camera_clip_plane_points(window_x, window_y)
        mn, mf = self.camera_view * (cn,cf)
        return mn, mf

    # Appears that Qt has disabled touch events on Mac due to unresolved scrolling lag problems.
    # Searching for qt setAcceptsTouchEvents shows they were disabled Oct 17, 2012.
    # A patch that allows an environment variable QT_MAC_ENABLE_TOUCH_EVENTS to allow touch
    # events had status "Review in Progress" as of Jan 16, 2013 with no more recent update.
    # The Qt 5.0.2 source code qcocoawindow.mm does not include the environment variable patch.
    def TODO_event(self, event):

        t = event.type()
        print ('event', int(t))
        if t == QtCore.QEvent.TouchBegin:
            print ('touch begin')
        elif t == QtCore.QEvent.TouchUpdate:
            print ('touch update')
        elif t == QtCore.QEvent.TouchEnd:
            print ('touch end')
        return QtOpenGL.QGLWidget.event(self, event)

    # Button is "left", "middle", or "right"
    def bind_mouse_mode(self, button, mouse_down,
                        mouse_drag = None, mouse_up = None):
        self.mouse_modes[button] = (mouse_down, mouse_drag, mouse_up)
        
    def mousePressEvent(self, event):
        self.dispatch_mouse_event(event, 0)
    def mouseMoveEvent(self, event):
        self.dispatch_mouse_event(event, 1)
    def mouseReleaseEvent(self, event):
        self.dispatch_mouse_event(event, 2)
    def wheelEvent(self, event):
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
        w,h = self.window_size
        cx, cy = event.x()-0.5*w, event.y()-0.5*h
        r2 = min(0.5*w,0.5*h)**2
        f2 = 0.8**2
        self.mouse_perimeter = (cx*cx + cy*cy > f2*r2)
        self.remember_mouse_position(event)

    def mouse_up(self, event):
        self.last_mouse_position = None

    def remember_mouse_position(self, event):
        self.last_mouse_position = QtCore.QPoint(event.pos())

    def mouse_pause_tracking(self):
        cp = self.mapFromGlobal(QtGui.QCursor.pos())
        w,h = self.window_size
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
        p, s = self.first_intercept(lp.x(), lp.y())
        from .gui import show_status
        show_status('Mouse over %s' % s.description() if s else '')

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
        self.rotate(axis, angle)

    def mouse_rotation(self, event):

        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5*math.sqrt(dx*dx+dy*dy)
        if self.mouse_perimeter:
            # z-rotation
            axis = (0,0,1)
            w, h = self.window_size
            ex, ey = event.x()-0.5*w, event.y()-0.5*h
            if -dy*ex+dx*ey < 0:
                angle = -angle
        else:
            axis = (dy,dx,0)
        return axis, angle

    def mouse_translate(self, event):

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        self.translate(psize*dx, -psize*dy, 0)

    def mouse_translate_molecules(self, event):

        mols = self.molecules()
        msel = [m for m in mols if m in self.selected]
        if msel:
            mols = msel
        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        self.translate(psize*dx, -psize*dy, 0, mols)

    def mouse_rotate_molecules(self, event):

        mols = self.molecules()
        msel = [m for m in mols if m in self.selected]
        if msel:
            mols = msel
        axis, angle = self.mouse_rotation(event)
        self.rotate(axis, angle, mols)

    def mouse_zoom(self, event):        

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        self.translate(0, 0, 3*psize*dy)

    def wheel_zoom(self, event):        

        d = event.angleDelta().y()/120.0   # Usually one wheel click is delta of 120
        psize = self.pixel_size()
        self.translate(0, 0, 100*d*psize)
        
    def mouse_contour_level(self, event):

        dx, dy = self.mouse_motion(event)
        f = -0.001*dy
        
        from ..VolumeViewer.volume import Volume
        for m in self.models:
            if isinstance(m, Volume):
                adjust_threshold_level(m, f)
                m.show()
        
    def wheel_contour_level(self, event):
        d = event.angleDelta().y()       # Usually one wheel click is delta of 120
        f = d/(120.0 * 30)
        for m in self.models:
            adjust_threshold_level(m, f)
            m.show()

    def rotate(self, axis, angle, models = None):

        self.update_center_of_rotation()
        # Rotation axis is in camera coordinates.
        # Center of rotation is in model coordinates.
        cv = self.camera_view
        from ..geometry import place
        maxis = cv.apply_without_translation(axis)
        r = place.rotation(maxis, angle, self.center_of_rotation)
        if models is None:
            self.set_camera_view(r.inverse()* cv)
        else:
            for m in models:
                m.place = r * m.place
        self.redraw_needed = True

    # Translation is in camera coordinates.  Sign is for moving models.
    def translate(self, dx, dy, dz, models = None):

        self.center_of_rotation_needs_update()
        cv = self.camera_view
        from ..geometry import place
        mt = cv.apply_without_translation((dx,dy,dz))
        t = place.translation(mt)
        if models is None:
            self.set_camera_view(t.inverse()*cv)
            n,f = self.near_far_clip
            self.near_far_clip = (n-dz,f-dz)
        else:
            for m in models:
                m.place = t * m.place
        self.redraw_needed = True

    def pixel_size(self, p = None):

        if p is None:
            p = self.center_of_rotation

        # Pixel size at center on near clip plane
        w,h = self.window_size
        from math import pi, tan
        fov = self.field_of_view * pi/180

        c = self.camera_position()
        from ..geometry import vector
        ps = vector.distance(c,p) * 2*tan(0.5*fov) / w
        return ps

#    def keyPressEvent(self, event):

#        from .shortcuts import keyboard_shortcuts as ks
#        ks.key_pressed(event)

    def maps(self):
        from ..VolumeViewer import Volume
        return tuple(m for m in self.models if isinstance(m,Volume))

    def molecules(self):
        from ..molecule import Molecule
        return tuple(m for m in self.models if isinstance(m,Molecule))
            
    def quit(self):
        import sys
        sys.exit(0)

# glFrustum() matrix
def frustum(left, right, bottom, top, zNear, zFar):
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = - (zFar + zNear) / (zFar - zNear)
    D = - (2 * zFar * zNear) / (zFar - zNear)
    E = 2 * zNear / (right - left)
    F = 2 * zNear / (top - bottom)
    m = ((E, 0, 0, 0),
         (0, F, 0, 0),
         (A, B, C, -1),
         (0, 0, D, 0))
    return m

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
        

def draw_image_text(qi, text, color = (255,255,255), bgcolor = None,
                    font_name = 'Helvetica', font_size = 24):
  p = QtGui.QPainter(qi)
  w,h = qi.width(), qi.height()

  while True and font_size > 6:
    f = QtGui.QFont(font_name, font_size)
    p.setFont(f)
    fm = p.fontMetrics()
    wt = fm.width(text)
    if wt <= w:
      break
    font_size = int(font_size * (w/wt))

  fh = fm.height()
  r = QtCore.QRect(0,h-fh,w,fh)
  if not bgcolor is None:
    p.fillRect(r, QtGui.QColor(*bgcolor))
  p.setPen(QtGui.QColor(*color))
  p.drawText(r, QtCore.Qt.AlignCenter, text)
