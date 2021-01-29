# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Command to view models in HTC Vive or Oculus Rift for ChimeraX.
#
def lookingglass(session, enable = None, device_number = 0,
                 view_angle = None, field_of_view = None, depth_offset = None,
                 verbose = False, quilt = False):
    '''
    Render to LookingGlass holographic display.
    '''
    
    if enable is None:
        enable = True

    lg_window = getattr(session, '_lg_window', None)
    if enable:
        if lg_window is None:
            lg_window = LookingGlassWindow(session,
                                           device_number = device_number,
                                           view_angle = view_angle,
                                           field_of_view = field_of_view,
                                           depth_offset = depth_offset,
                                           verbose = verbose, quilt = quilt)
            session._lg_window = lg_window
            _bind_depth_mouse_mode(session)
        else:
            lg_camera = lg_window.looking_glass_camera
            if view_angle is not None:
                lg_camera.view_angle = view_angle
            if field_of_view is not None:
                lg_camera.field_of_view = field_of_view
            if depth_offset is not None:
                lg_camera.depth_offset = depth_offset
                
    elif not enable and lg_window:
        lg_window.delete()
        delattr(session, '_lg_window')
        
# -----------------------------------------------------------------------------
#
def register_lookingglass_command(logger):
    from chimerax.core.commands import register, create_alias, CmdDesc, BoolArg, IntArg, FloatArg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('device_number', IntArg),
                              ('view_angle', FloatArg),
                              ('field_of_view', FloatArg),
                              ('depth_offset', FloatArg),
                              ('verbose', BoolArg),
                              ('quilt', BoolArg),
                   ],
                   synopsis = 'Render to LookingGlass holographic display.',
                   url = 'help:user/commands/device.html#lookingglass')
    register('lookingglass', desc, lookingglass, logger=logger)
    create_alias('device lookingglass', 'lookingglass $*', logger=logger,
                 url='help:user/commands/device.html#lookingglass')

# -----------------------------------------------------------------------------
#
from chimerax.graphics import Camera
class LookingGlassCamera(Camera):

    always_draw = True	# Draw even if main window iconified.
    name = 'lookingglass'
    
    def __init__(self, session, device_number = 0,
                 view_angle = None, field_of_view = 22,
                 depth_offset = 0, verbose = False):

        Camera.__init__(self)

        self._session = session
        self._device_number = device_number
        self._hpc = hpc = self._load_device_parameters()	# Parameters needed by shader

        if verbose:
            session.logger.info('HoloPlay library info:\n%s' % hpc.info())

        self.field_of_view = field_of_view	# Corresponds to recommended 14 degree vertical field.
        if view_angle is None:
            view_angle = hpc.hpc_GetDeviceViewCone(device_number) if self.found_device() else 40
        self._view_angle = view_angle		# Camera range of x positions
        self._depth_offset = depth_offset	# Depth shift of center of bounding box from mid-depth
        self._focus_depth = 100			# Scene units
        self._update_focus_depth()
        self._quilt_size = (w,h) = (4096, 4096)
        self._quilt_columns = c = 5
        self._quilt_rows = r = 9
        self._quilt_tile_size = (w//c, h//r)
        self._shader = None			# Shader for rendering quilt to window
        self._framebuffer = None		# For rendering into quilt texture
        self._texture_drawing = None		# For rendering quilt to window
        self._show_quilt = False

    def delete(self):
        self._hpc.hpc_CloseApp()
        self._hpc = None

        fb = self._framebuffer
        if fb:
            fb.delete(make_current = True)
        self._framebuffer = None

        d = self._texture_drawing
        if d:
            d.delete()
        self._texture_drawing = None
        
        self._session = None

    def found_device(self):
        return self._device_number < self._hpc.hpc_GetNumDevices()
        
    def screen_name(self):
        return self._hpc.hpc_GetDeviceHDMIName(self._device_number) if self.found_device() else None

    def _get_field_of_view(self):
        return self._field_of_view
    def _set_field_of_view(self, field_of_view):
        self._field_of_view = field_of_view
        self._session.main_view.camera.field_of_view = field_of_view
        self._redraw_needed()
    field_of_view = property(_get_field_of_view, _set_field_of_view)

    def _get_view_angle(self):
        return self._view_angle
    def _set_view_angle(self, view_angle):
        self._view_angle = view_angle
        self._redraw_needed()
    view_angle = property(_get_view_angle, _set_view_angle)

    def _redraw_needed(self):
        self._session.main_view.redraw_needed = True
        
    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            xoffset = self._x_offset_scene(view_num)
            from chimerax.geometry import place
            t = place.translation((xoffset, 0, 0))
            v = camera_position * t
        return v

    def _x_offset_scene(self, view_num):
        from math import tan, pi
        xrange = self._focus_depth * 2*tan(0.5 * self._view_angle * pi / 180)
        x = xrange * (view_num / (self.number_of_views()-1) - 0.5)
        return x

    def _x_offset_pixels(self, view_num):
        xos = self._x_offset_scene(view_num)
        from math import tan, pi
        wscene = self._focus_depth * 2*tan(0.5 * self._field_of_view * pi / 180)
        wpixels = self._quilt_tile_size[0]
        xop = wpixels * xos / wscene
        return xop

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        return self._quilt_rows * self._quilt_columns

    def view_width(self, point):
        from chimerax.graphics.camera import perspective_view_width
        return perspective_view_width(point, self.position.origin(), self._field_of_view)

    def view_all(self, bounds, window_size = None, pad = 0):
        from chimerax.graphics.camera import perspective_view_all
        self.position = perspective_view_all(bounds, self.position, self._field_of_view, window_size, pad)
        self._update_focus_depth()

    def _set_position(self, p):
        Camera.set_position(self, p)
        self._update_focus_depth()
    position = property(Camera.position.fget, _set_position)

    def _get_depth_offset(self):
        return self._depth_offset
    def _set_depth_offset(self, offset):
        self._depth_offset = offset
        self._update_focus_depth()
        self._redraw_needed()
    depth_offset = property(_get_depth_offset, _set_depth_offset)
        
    def _update_focus_depth(self):
        v = self._session.main_view
        b = v.drawing_bounds()
        if b is None:
            return
        view_dir = -self.position.z_axis()
        delta = b.center() - self.position.origin()
        from chimerax.geometry import inner_product
        d = inner_product(delta, view_dir)
        d -= self._depth_offset
        if d != self._focus_depth:
            self._focus_depth = d
            self._redraw_needed()

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        pixel_shift = (self._x_offset_pixels(view_num), 0)
        from chimerax.graphics.camera import perspective_projection_matrix
        return perspective_projection_matrix(self._field_of_view, window_size,
                                             near_far_clip, pixel_shift)
    
    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        if view_num == 0:
            fb = self._quilt_framebuffer(render)
            render.push_framebuffer(fb)
            render.draw_background()
        qc = self._quilt_columns
        x, y = (view_num % qc), (view_num // qc)
        w, h  = self._quilt_tile_size
        render.set_viewport(x * w, y * h, w, h)

    def draw_background(self, view_num, render):
        if render.current_framebuffer() is not self._quilt_framebuffer(render):
            render.draw_background()
    
    def combine_rendered_camera_views(self, render):
        render.pop_framebuffer()
        self._draw_quilt(render)

    def _quilt_framebuffer(self, render):
        fb = self._framebuffer
        if fb is None:
            from chimerax.graphics import Texture, opengl
            t = Texture()
            qw,qh = self._quilt_size
            t.initialize_rgba((qw,qh))
            fb = opengl.Framebuffer('LookingGlass quilt', render.opengl_context, color_texture = t)
            self._framebuffer = fb
        return fb

    def _load_device_parameters(self):
        from .holoplay import HoloPlayCore
        hpc = HoloPlayCore()
        err = hpc.hpc_InitializeApp('ChimeraX', hpc.hpc_LICENSE_NONCOMMERCIAL)
        if err:
            msg = 'Failed to initialize HoloPlay: %s' % hpc.error_code_message(err)
            self._session.logger.warning(msg)
        return hpc
    
    def _draw_quilt(self, render):
        drawing = self._quilt_drawing()
        ds = drawing._draw_shape
        ds.activate_bindings(render)
        self._activate_quilt_shader(render)
        render.enable_depth_test(False)
        t = drawing.texture
        t.bind_texture()
        ds.draw(drawing.Solid)  # draw rectangle
        t.unbind_texture()
        render.enable_depth_test(True)

    def _activate_quilt_shader(self, render):
        shader = self._shader
        if shader is None:
            qsize = (self._quilt_size[0], self._quilt_size[1], self._quilt_rows, self._quilt_columns)
            shader = self._hpc.quilt_shader(self._device_number, qsize, self._show_quilt)
            self._shader = shader
        from OpenGL import GL
        GL.glUseProgram(shader.program_id)
        render._opengl_context.current_shader_program = None   # Clear cached shader
        shader.set_integer("screenTex", 0)    # Texture unit 0.
        self._hpc.set_shader_uniforms(shader)

    def _quilt_drawing(self):
        '''Used  to render ChimeraX desktop graphics window.'''
        td = self._texture_drawing
        if td is None:
            # Drawing for rendering quilt texture to ChimeraX window
            texture = self._framebuffer.color_texture
            from chimerax.graphics.drawing import _texture_drawing
            self._texture_drawing = td = _texture_drawing(texture)
            td.opaque_texture = True
            td._create_vertex_buffers()
            td._update_buffers()
        return td
    
# -------------------------------------------------------------------------------------------------
#
from Qt.QtGui import QWindow
class LookingGlassWindow(QWindow):
    def __init__(self, session, device_number = 0,
                 view_angle = None, field_of_view = None, depth_offset = None,
                 verbose = False, quilt = False):
        self._session = session

        # Create camera for rendering LookingGlass image
        cam_settings = {name:value for name, value in (
                         ('device_number', device_number),
                         ('view_angle', view_angle),
                         ('field_of_view', field_of_view),
                         ('depth_offset', depth_offset),
                         ('verbose', verbose))
                        if value is not None}
        lgc = LookingGlassCamera(session, **cam_settings)
        self.looking_glass_camera = lgc

        # Create fullscreen window on LookingGlass display
        screen = None if quilt else self._looking_glass_screen()
        QWindow.__init__(self, screen = screen)
        from Qt.QtGui import QSurface
        self.setSurfaceType(QSurface.OpenGLSurface)

        if screen:
            from sys import platform
            if platform == 'win32':
                # Qt 5.12 hangs if OpenGL window is put on second display
                # but works if moved after a delay.
                self.setScreen(self._session.ui.primaryScreen())
                def _set_fullscreen(self=self, screen=screen):
                    self.setScreen(screen)
                    self.showFullScreen()
                # Have to save reference to timer or it is deleted before executing.
                self._timer = self._session.ui.timer(1000, _set_fullscreen)
                #from Qt.QtCore import Qt
                #self.setFlags(Qt.FramelessWindowHint)
                self.show()
            else:
                self.showFullScreen()
        else:
            lgc._show_quilt = True
            self.setWidth(500)
            self.setHeight(500)
            self.show()

        t = session.triggers
        self._render_handler = t.add_handler('frame drawn', self._frame_drawn)
        self._app_quit_handler = t.add_handler('app quit', self._app_quit)
        
    def _looking_glass_screen(self):
        screen_name = self.looking_glass_camera.screen_name()
        if screen_name is None:
            screen = None
        else:
            app = self._session.ui
            screens = app.screens()
            lkg_screens = [s for s in screens if s.name() == screen_name]
            screen = lkg_screens[0] if lkg_screens else None
            if screen is None:
                from sys import platform
                if platform == 'win32' or platform == 'linux':
                    # On Windows 10 and Linux Ubuntu 18.04 with Qt 5.12 the Qt screen name
                    # does not match the HoloPlay screen name.
                    # There appears to be no way to reliably identify the correct screen.
                    # Use the non-primary screen.
                    extra_screens = [s for s in screens if s is not app.primaryScreen()]
                    if len(extra_screens) == 1:
                        screen = extra_screens[0]

        if screen is None:
            if screen_name is None:
                msg = 'Did not find any connected LookingGlass display'
            else:
                msg = ('Did not find LookingGlass screen name %s, found %s'
                       % (screen_name, ', '.join(s.name() for s in screens)))
            self._session.logger.warning(msg)
        return screen

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._render_handler)
        t.remove_handler(self._app_quit_handler)
        self._render_handler = None
        self._app_quit_handler = None
        self.looking_glass_camera.delete()
        self.looking_glass_camera = None
        self.close()	# QWindow method

    def _app_quit(self, tname, tdata):
        self.delete()

    def event(self, event):
        from Qt.QtCore import QEvent
        if event.type() == QEvent.Expose:
            self._render()
        return QWindow.event(self, event)

    def _frame_drawn(self, tname, tdata):
        self._render()
        
    def _render(self):
        camera = self.looking_glass_camera
        if camera is None:
            return   # Window deleted
        view = self._session.main_view
        r = view.render
        mvwin = r.use_shared_context(self)
        camera.position = view.camera.position
        redraw = view.redraw_needed
        view.draw(camera = camera)
        view.redraw_needed = redraw  # Restore redraw needed flag since main window needs redraw.
#        self._save_screen_image(r, view)
        r.use_shared_context(mvwin)  # Reset opengl window
        r.done_current()

    def _save_screen_image(self, r, view):
        self._save_screen_image_countdown = getattr(self, '_save_screen_image_countdown', 50) - 1
        if self._save_screen_image_countdown == 0:
            s = self.size()
            w, h = s.width(), s.height()
            print ('capturing screen window size', w, h)
            view._use_opengl()
            rgba = r.frame_buffer_image(w, h, front_buffer = True)
            from PIL import Image
            Image.fromarray(rgba[::-1]).save('screen.png', 'PNG')

from chimerax.mouse_modes import MouseMode
class DepthShiftMouseMode(MouseMode):
    '''
    Mouse mode to move LookingGlass focal plane.
    '''
    name = 'lookingglass depth'
    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.speed = 1

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        delta_z = 2*psize*dy*self.speed
        self._depth_shift(delta_z)

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        delta_z = 20*d*psize*self.speed
        self._depth_shift(delta_z)

    def _depth_shift(self, delta_z):
        lg_window = getattr(self.session, '_lg_window', None)
        if lg_window:
            lg_camera = lg_window.looking_glass_camera
            lg_camera.depth_offset -= delta_z

def _bind_depth_mouse_mode(session):
    mm = session.ui.mouse_modes
    mode = mm.named_mode(DepthShiftMouseMode.name)
    if mode is None:
        mode = DepthShiftMouseMode(session)
        mm.add_mode(mode)
    mm.bind_mouse_mode(mouse_button = 'wheel', mouse_modifiers = ['shift'], mode = mode)
