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
def vr(session, enable = None, room_position = None, mirror = True, icons = False,
       show_controllers = True, multishadow_allowed = False):
    '''Enable stereo viewing and head motion tracking with virtual reality headsets using SteamVR.

    Parameters
    ----------
    enable : bool
      Enable or disable use of an HTC Vive headset or Oculus Rift headset using SteamVR.
      The device must be connected
      and powered on to enable it. Graphics will not be updated in the main
      ChimeraX window because the different rendering rates of the headset and a
      conventional display will cause stuttering of the headset graphics.
      Also the Side View panel in the main ChimeraX window should be closed to avoid
      stuttering.
    room_position : Place or "report"
      Maps physical room coordinates to molecular scene coordinates.
      Room coordinates have origin at center of room and units are meters.
    mirror : bool
      Whether to update the ChimeraX graphics window.  This also turns off waiting
      for display vertical sync on the computer monitor so that the 60 Hz refresh rate
      does not slow down the 90 Hz rendering to the VR headset.
    icons : bool
      Whether to show a panel of icons when controller trackpad is touched.
      For demonstrations the icons can be too complex and it is better not to have icons.
      Default false.
    show_controllers : bool
      Whether to show the hand controllers in the scene. Default true.
    multishadow_allowed : bool
      If this option is false and multi-shadow lighting is enabled (ambient occlusion) when vr is
      enabled, then lighting is switched to simple lighting.  If the option is true then no
      changes to lighting mode are made.  Often rendering is not fast enough
      to support multishadow lighting so this option makes sure it is off so that stuttering
      does not occur.  Default False.
    '''
    
    if enable is None and room_position is None:
        enable = True

    if enable is not None:
        if enable:
            start_vr(session, multishadow_allowed)
        else:
            stop_vr(session)

    v = session.main_view
    c = v.camera
    if room_position is not None:
        if not isinstance(c, SteamVRCamera):
            from chimerax.core.errors import UserError
            raise UserError('Cannot use vr roomPosition unless vr enabled.')
        if isinstance(room_position, str) and room_position == 'report':
            p = ','.join('%.5g' % x for x in tuple(c.room_to_scene.matrix.flat))
            session.logger.info(p)
        else:
            c.room_to_scene = room_position
            c._last_position = c.position

    if isinstance(c, SteamVRCamera):
        if mirror is not None:
            c.mirror_display = mirror
            wait_for_vsync(session, mirror)
        if show_controllers is not None:
            for hc in c.hand_controllers(show_controllers):
                hc.show_in_scene(show_controllers)
        if icons is not None: 
            for hc in c.hand_controllers():
                hc.use_icons = icons
            
# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, NoArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('mirror', BoolArg),
                              ('icons', BoolArg),
                              ('show_controllers', BoolArg),],
                   synopsis = 'Start SteamVR virtual reality rendering')
    register('device vr', desc, vr, logger=logger)
    create_alias('vr', 'device vr $*', logger=logger)

# -----------------------------------------------------------------------------
#
def start_vr(session, multishadow_allowed = False):

    v = session.main_view
    if not multishadow_allowed and v.lighting.multishadow > 0:
        from chimerax.core.commands import run
        run(session, 'lighting simple')

    if isinstance(v.camera, SteamVRCamera):
        return

    try:
        import openvr
    except Exception as e:
        from chimerax.core.errors import UserError
        raise UserError('Failed to import OpenVR module: %s' % str(e))
    
    v.camera = SteamVRCamera(session)
    # Set redraw timer to redraw as soon as Qt events processsed to minimize dropped frames.
    session.ui.main_window.graphics_window.set_redraw_interval(0)

    msg = 'started SteamVR rendering'
    log = session.logger
    log.status(msg)
    log.info(msg)

# -----------------------------------------------------------------------------
#
def stop_vr(session):

    c = session.main_view.camera
    if isinstance(c, SteamVRCamera):
        # Have to delay shutdown of SteamVR connection until draw callback
        # otherwise it clobbers the Qt OpenGL context making entire gui black.
        def replace_camera(s = session):
            from chimerax.core.graphics import MonoCamera
            v = s.main_view
            v.camera = MonoCamera()
            s.ui.main_window.graphics_window.set_redraw_interval(10)
            v.view_all()
        c.close(replace_camera)
        wait_for_vsync(session, True)

# -----------------------------------------------------------------------------
#
def wait_for_vsync(session, mirror):
    r = session.main_view.render
    r.make_current()
    if not r.wait_for_vsync(not mirror):
        if mirror:
            session.logger.warning('Mirror may cause VR stutter.'
                                   '  Could not turn off wating for vsync on main display.')

# -----------------------------------------------------------------------------
#
from chimerax.core.graphics import Camera
class SteamVRCamera(Camera):

    def __init__(self, session):

        Camera.__init__(self)

        self._session = session
        self._framebuffer = None	# For rendering each eye view to a texture
        self._texture_drawing = None	# For mirror display
        self._last_position = None
        self._last_h = None
        self._close = False
        self._controller_models = []	# List of HandControllerModel
        self.user_interface = UserInterface(self, session)
        self.mirror_display = False	# Mirror right eye in ChimeraX window
        				# This causes stuttering in the Vive.

        self.room_position = None	# Camera position in room coordinates

        import openvr
        self.vr_system = vrs = openvr.init(openvr.VRApplication_Scene)

        self._render_size = self.vr_system.getRecommendedRenderTargetSize()
        self.compositor = openvr.VRCompositor()
        if self.compositor is None:
            raise RuntimeError("Unable to create compositor") 

        # Compute projection and eye matrices, units in meters
        zNear = 0.1
        zFar = 500.0
        # TODO: Scaling models to be huge causes clipping at far clip plane.

        # Left and right projections are different. OpenGL 4x4.
        pl = vrs.getProjectionMatrix(openvr.Eye_Left, zNear, zFar)
        self.projection_left = hmd44_to_opengl44(pl)
        pr = vrs.getProjectionMatrix(openvr.Eye_Right, zNear, zFar)
        self.projection_right = hmd44_to_opengl44(pr)

        # Eye shifts from hmd pose.
        vl = vrs.getEyeToHeadTransform(openvr.Eye_Left)
        self.eye_shift_left = hmd34_to_position(vl)
        vr = vrs.getEyeToHeadTransform(openvr.Eye_Right)
        self.eye_shift_right = hmd34_to_position(vr)

        # Map ChimeraX scene coordinates to OpenVR room coordinates
        self.fit_scene_to_room()
        
        # Update camera position every frame.
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self._poses = poses_t()
        h = session.triggers.add_handler('new frame', self.next_frame)
        self._new_frame_handler = h

    def _get_position(self):
        return Camera.get_position(self)
    def _set_position(self, position):
        '''Move camera in scene while keeping camera in a fixed position in room.'''
        self.room_to_scene =  position * self.position.inverse() * self.room_to_scene
        Camera.set_position(self, position)
        ui = self.user_interface
        if ui.shown():
            ui.move()
    position = property(_get_position, _set_position)

    def _move_camera_in_room(self, position):
        '''Move camera to given scene position without changing scene position in room.'''
        Camera.set_position(self, position)
        
    def fit_scene_to_room(self,
                          scene_bounds = None,
                          room_scene_size = 2, 		# Initial virtual model size in meters
                          room_center = (0,1,0),
                          ):
        '''Set transform relating scene coordinates and room coordinates.'''
# Chaperone bounds reported as -2 to 2 in x, -1.2 to 1.2 in z, 0 in y (floor).
# x is -2 near vive computer, +2 near vizvault door.
# z is 1.2 near door and vive computer, and -1.2 on opposite wall.
# y is 0 near floor and 2.5 near ceiling.
#        chaperone = openvr.VRChaperone()
#        result, rect = chaperone.getPlayAreaRect()
#        for c in rect.vCorners:
#            print('corners', tuple(c.v))
        from numpy import array, zeros, float32
        b = scene_bounds
        if b is None:
            # TODO: Avoid this undisplay hack used to eliminate controllers from bounds.
            cm = self._controller_models
            hcd = [hc.display for hc in cm]
            for hc in cm:
                hc.display = False
            b = self._session.main_view.drawing_bounds()
            for hc, disp in zip(cm, hcd):
                hc.display = disp
        if b:
            scene_size = b.width()
            scene_center = b.center()
        else:
            scene_size = 1
            scene_center = zeros((3,), float32)
        # First apply scene shift then scene scale to get room coords
        from chimerax.core.geometry import translation, scale
        self.room_to_scene = (translation(scene_center) *
                              scale(scene_size/room_scene_size) *
                              translation(-array(room_center, float32)))

    def move_scene(self, move):
        self.room_to_scene = self.room_to_scene * move
        ui = self.user_interface
        if ui.shown():
            ui.move()

    def close(self, close_cb = None):
        self._close = True
        self._close_cb = close_cb
        self._session.main_view.redraw_needed = True
        
    def _delayed_close(self):
        # Apparently OpenVR doesn't make its OpenGL context current
        # before deleting resources.  If the Qt GUI opengl context is current
        # openvr deletes the Qt resources instead.  So delay openvr close
        # until after rendering so that openvr opengl context is current.
        self._session.triggers.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
        for hc in self._controller_models:
            hc.close()
        self._controller_models = []
        self.user_interface.close()
        import openvr
        openvr.shutdown()
        self.vr_system = None
        self.compositor = None
        fb = self._framebuffer
        if fb is not None:
            self._session.main_view.render.make_current()
            fb.delete()
            self._framebuffer = None
        if self._close_cb:
            self._close_cb()	# Replaces the main view camera and resets redraw rate.

        
    def name(self):
        '''Name of camera.'''
        return 'vr'

    def next_frame(self, *_):
        c = self.compositor
        if c is None or self._close:
            return
        import openvr
        c.waitGetPoses(self._poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
        hmd_pose0 = self._poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not hmd_pose0.bPoseIsValid:
            return
        # head to room coordinates.
        H = hmd34_to_position(hmd_pose0.mDeviceToAbsoluteTracking)

        self.process_controller_events()
        
        # Compute camera scene position from HMD position in room
        from chimerax.core.geometry import scale
        S = scale(self.scene_scale)
        C, last_C = self.position, self._last_position
        if last_C is not None and C is not last_C:
            # Camera moved by mouse or command.
            hs = self._last_h * S
            self.room_to_scene = C * hs.inverse()
        self.room_position = rp = H * S
        Cnew = self.room_to_scene * rp
        self._last_position = Cnew
        self._move_camera_in_room(Cnew)
        self._last_h = H

        self._session.triggers.activate_trigger('vr update', self)

    @property
    def scene_scale(self):
        '''Scale factor from scene to room coordinates.'''
        x,y,z = self.room_to_scene.matrix[:,0]
        from math import sqrt
        return 1/sqrt(x*x + y*y + z*z)
    
    def process_controller_events(self):

        self.process_controller_buttons()
        self.process_controller_motion()

    def process_controller_buttons(self):
        
        # Check for button press
        vrs = self.vr_system
        import openvr
        e = openvr.VREvent_t()
        while vrs.pollNextEvent(e):
            for hc in self.hand_controllers():
                hc.process_event(e, self)

        # Touchpad motion does not generate an event.
        for hc in self.hand_controllers():
            if hc.use_icons:
                hc.process_touchpad_motion()
                
    def process_controller_motion(self):

        self.check_if_controller_models_closed()
        for hc in self.hand_controllers():
            hc.process_motion(self)
        
    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            es = self.eye_shift_left if view_num == 0 else self.eye_shift_right
            t = es.scale_translation(1/self.scene_scale)
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        return 2

    def view_width(self, point):
        fov = 100	# Effective field of view, degrees
        from chimerax.core.graphics.camera import perspective_view_width
        return perspective_view_width(point, self.position.origin(), fov)

    def view_all(self, bounds, window_size = None, pad = 0):
        fov = 100	# Effective field of view, degrees
        from chimerax.core.graphics.camera import perspective_view_all
        p = perspective_view_all(bounds, self.position, fov, window_size, pad)
        self._move_camera_in_room(p)
        self._last_position = None
        self._last_h = None
        self.fit_scene_to_room(bounds)

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        if view_num == 0:
            p = self.projection_left
        elif view_num == 1:
            p = self.projection_right
        pm = p.copy()
        pm[:3,:] *= self.scene_scale
        return pm

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        fb = self._texture_framebuffer()
        if view_num == 0:
            render.push_framebuffer(fb)
        elif view_num == 1:
            if not self._close:
                # Submit left eye texture (view 0) before rendering right eye (view 1)
                import openvr
                result = self.compositor.submit(openvr.Eye_Left, fb.openvr_texture)
                self._check_for_compositor_error('left', result, render)

    def _check_for_compositor_error(self, eye, result, render):
        import openvr
        if result != openvr.VRCompositorError_None:
            self._session.logger.info('SteamVR compositor submit for %s eye returned error %d'
                                      % (eye, result))
        err_msg = render.check_for_opengl_errors()
        if err_msg:
            self._session.logger.info('SteamVR compositor submit for %s eye produced an OpenGL error "%s"'
                                      % (eye, err_msg))

    def combine_rendered_camera_views(self, render):
        '''
        Submit right eye texture image to OpenVR. Left eye was already submitted
        by set_render_target() when render target switched to right eye.
        '''
        fb = render.pop_framebuffer()

        if not self._close:
            import openvr
            result = self.compositor.submit(openvr.Eye_Right, fb.openvr_texture)
            self._check_for_compositor_error('right', result, render)

        if self.mirror_display:
            # Render right eye to ChimeraX window.
            from chimerax.core.graphics.drawing import draw_overlays
            draw_overlays([self._mirror_drawing()], render)

        if self._close:
            self._delayed_close()

    def _texture_framebuffer(self):

        tw,th = self._render_size
        fb = self._framebuffer
        if fb is None or fb.width != tw or fb.height != th:
            from chimerax.core.graphics import Texture, opengl
            t = Texture()
            t.initialize_rgba((tw,th))
            self._framebuffer = fb = opengl.Framebuffer(color_texture = t)
            # OpenVR texture id object
            import openvr
            fb.openvr_texture = ovrt = openvr.Texture_t()
            from ctypes import c_void_p
            ovrt.handle = c_void_p(int(t.id))
            ovrt.eType = openvr.TextureType_OpenGL
            ovrt.eColorSpace = openvr.ColorSpace_Gamma
        return fb

    def _mirror_drawing(self):
        '''Only used for mirror headset view to ChimeraX graphics window.'''
        td = self._texture_drawing
        if td is None:
            # Drawing object for rendering to ChimeraX window
            from chimerax.core.graphics.drawing import _texture_drawing
            t = self._framebuffer.color_texture
            self._texture_drawing = td = _texture_drawing(t)
            td.opaque_texture = True
        return td

    def do_swap_buffers(self):
        return self.mirror_display

    def hand_controllers(self, show = True):
        cm = self._controller_models
        if len(cm) == 0:
            # TODO: If controller is turned on after initialization then it does not get in list.
            import openvr
            controller_ids = [d for d in range(openvr.k_unMaxTrackedDeviceCount)
                              if self.vr_system.getTrackedDeviceClass(d)
                                 == openvr.TrackedDeviceClass_Controller]
            ses = self._session
            vrs = self.vr_system
            cm = [HandControllerModel(d, ses, vrs, show) for d in controller_ids]
            self._controller_models = cm
        return cm

    def other_controller(self, controller):
        for hc in self.hand_controllers():
            if hc != controller:
                return hc
        return None

    def check_if_controller_models_closed(self):
        cm = self._controller_models
        cma =[hc for hc in cm if not hc.deleted]
        if len(cma) < len(cm):
            self._controller_models = cma

class UserInterface:
    '''
    Panel in VR showing ChimeraX main window.
    Buttons can be clicked with hand controllers.
    '''
    def __init__(self, camera, session):
        self._camera = camera
        self._session = session
        self._width = 1		# Billboard width in room coords, meters.
        self._height = None	# Height in room coords determined by window aspect and width.
        self._window_size = None # Window size in pixels
        self._ui_click_range = 0.05 # Maximum distance of click from plane, room coords, meters.
        self._ui_drawing = None
        self._start_ui_move_time = None
        self._last_ui_position = None
        self._ui_hide_time = 0.3	# seconds. Max application button press/release time to hide ui

    def close(self):
        ui = self._ui_drawing
        if ui:
            self._session.models.close([ui])
            self._ui_drawing = None

    def shown(self):
        ui = self._ui_drawing
        return ui is not None and ui.display
    
    def show(self, room_position):
        ui = self._ui_drawing
        if ui is None:
            self._ui_drawing = ui = self._create_ui_drawing()
        self._update_ui_image(ui)
        ui.room_position = room_position
        ui.position = self._camera.room_to_scene * room_position
        ui.display = True

    def move(self, room_motion = None):
        ui = self._ui_drawing
        if ui and ui.display:
            if room_motion:
                ui.room_position = room_motion * ui.room_position
            ui.position = self._camera.room_to_scene * ui.room_position            
        
    def hide(self):
        ui = self._ui_drawing
        if ui is not None:
            ui.display = False
            
    def click(self, pressed, room_point):
        if not self.shown():
            return False
        ui = self._ui_drawing
        x,y,z = ui.room_position.inverse() * room_point
        hw, hh = 0.5*self._width, 0.5*self._height
        cr = self._ui_click_range
        if x < -hw or x > hw or y < -hh or y > hh or z < -cr or z > cr:
            if pressed:
                return False # click not on billboard
        sx, sy = self._window_size
        px, py = sx * (x + hw) / (2*hw), sy * (hh - y) / (2*hh)

        if pressed:
            mmode = self._clicked_mouse_mode(px, py)
            if mmode:
                return mmode
        
        if not self._post_mouse_event(pressed, px, py):
            return False

        # TODO: Delay ui update until user interface echoes command.
        self._update_ui_image(ui)
        return True

    def _clicked_mouse_mode(self, window_x, window_y):
        w, pos = self._clicked_widget(window_x, window_y)
        from PyQt5.QtWidgets import QToolButton
        if isinstance(w, QToolButton):
            a = w.defaultAction()
            if hasattr(a, 'mouse_mode'):
                return a.mouse_mode
        return None
    
    def _post_mouse_event(self, pressed, window_x, window_y):
        w, pos = self._clicked_widget(window_x, window_y)
        if w is None or pos is None:
            return False
        from PyQt5.QtGui import QMouseEvent
        from PyQt5.QtCore import Qt, QEvent
        type = QEvent.MouseButtonPress if pressed else QEvent.MouseButtonRelease
        buttons = Qt.LeftButton if pressed else Qt.NoButton
        me = QMouseEvent(type, pos, Qt.LeftButton, buttons, Qt.NoModifier)
        self._session.ui.postEvent(w, me)
        return True
        
    def _clicked_widget(self, window_x, window_y):
        ui = self._session.ui
        mw = ui.main_window
        from PyQt5.QtCore import QPoint, QPointF
        gp = mw.mapToGlobal(QPoint(int(window_x), int(window_y)))
        # Mouse events sent to main window are not handled.  Need to send to widget under mouse.
        w = ui.widgetAt(gp)
        wpos = QPointF(w.mapFromGlobal(gp)) if w else None
        return w, wpos

    def _create_ui_drawing(self):
        ses = self._session
        from chimerax.core.models import Model
        m = Model('vr user interface', ses)
        ses.models.add([m])
        return m

    def _update_ui_image(self, model):
        ses = self._session
        im = ses.ui.window_image()
        from chimerax.core.graphics.drawing import rgba_drawing, qimage_to_numpy
        rgba = qimage_to_numpy(im)
        h,w = rgba.shape[:2]
        self._window_size = (w, h)
        aspect = h/w
        rw = self._width		# Billboard width in room coordinates
        self._height = rh = aspect * rw
        ses.main_view.render.make_current()	# Required OpenGL context for replacing texture.
        rgba_drawing(model, rgba, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))

    def display_ui(self, button_pressed, hand_room_position):
        if button_pressed:
            rp = hand_room_position
            self._last_ui_position = rp
            if self.shown():
                from time import time
                self._start_ui_move_time = time()
            else:
                # Orient horizontally and perpendicular to floor
                fx,fy,fz = rp.z_axis()
                from chimerax.core.geometry import orthonormal_frame
                p = orthonormal_frame((fx,0,fz), (0,1,0), origin = rp.origin())
                self.show(p)
        else:
            # End UI move, or hide.
            stime = self._start_ui_move_time
            from time import time
            if stime is not None and time() < stime + self._ui_hide_time:
                self.hide()
            self._start_ui_move_time = None

    def move_ui(self, hand_room_position):
        if self._start_ui_move_time is None:
            return
        luip = self._last_ui_position
        rp = hand_room_position
        self.move(rp * luip.inverse())
        self._last_ui_position = rp
        
from chimerax.core.models import Model
class HandControllerModel(Model):
    casts_shadows = False
    pickable = False
    _controller_colors = ((200,200,0,255), (0,200,200,255))
    SESSION_SAVE = False

    def __init__(self, device_index, session, vr_system, show = True, size = 0.20, aspect = 0.2):
        name = 'Hand %s' % device_index
        Model.__init__(self, name, session)
        self.device_index = device_index
        self.vr_system = vr_system
        self.use_icons = False
        self._mode = 'move scene'
        self._drag = False		# Whether trigger button is held down
        self._pose = None
        self._previous_pose = None
        self._zoom_center = None
        self._app_button_down = False
        self._icon_drawing = None
        self._icon_highlight_drawing = None
        self._icon_size = 128  # pixels
        self._icon_columns = 0
        self._icon_rows = 0
        self._icon_shortcuts = []
        self._icons_shown = False
        self._mouse_mode = None		# MouseMode for hand controller clicks
        self._laser_range = 5		# Range for mouse mode laser clicks
        self._ui_pressed = False	# Remember if click was on ui to make sure ui release event generated
        self._last_drag_room_position = None # Hand controller position at last drag_3d call
        
        self.room_position = None	# Hand controller position in room coordinates.

        from chimerax.core.surface.shapes import cone_geometry
        va, na, ta = cone_geometry(nc = 50, points_up = False)
        va[:,:2] *= aspect
        va[:,2] += 0.5		# Move tip to 0,0,0 for picking
        va *= size
        self._cone_vertices = va
        self._last_cone_scale = 1
        self._cone_length = size
        self.geometry = va, ta
        self.normals = na
        cc = self._controller_colors
        rgba8 = cc[device_index%len(cc)]
        from numpy import array, uint8
        self.color = array(rgba8, uint8)

        self._shown_in_scene = show
        if show:
            session.models.add([self])

    def close(self):
        if self._shown_in_scene:
            self.session.models.close([self])
        else:
            self.delete()

    def show_in_scene(self, show):
        models = self.session.models
        if show and not self._shown_in_scene:
            models.add([self])
        elif not show and self._shown_in_scene:
            models.remove([self])
        self._shown_in_scene = show
        
    def _update_position(self, camera):
        '''Move hand controller model to new position.
        Keep size constant in physical room units.'''
        dp = camera._poses[self.device_index].mDeviceToAbsoluteTracking
        self.room_position = self._pose = rp = hmd34_to_position(dp)
        if self._shown_in_scene:
            from chimerax.core.geometry import scale
            s = camera.scene_scale
            self.position = camera.room_to_scene * rp * scale(s)
            if s < 0.999*self._last_cone_scale or s > 1.001*self._last_cone_scale:
                self.vertices = (1/s) * self._cone_vertices
                self._last_cone_scale = s

    def tip_position(self):
        return self.scene_position.origin()

    def process_event(self, e, camera):
        
        t = e.eventType
        import openvr
        if (self.use_icons and
            (t == openvr.VREvent_ButtonTouch or
             t == openvr.VREvent_ButtonUntouch) and
            e.data.controller.button == openvr.k_EButton_SteamVR_Touchpad and
            e.trackedDeviceIndex == self.device_index):
            touch = (t == openvr.VREvent_ButtonTouch)
            if touch:
                xy = self.touchpad_position()
                if xy is not None:
                    self.show_icons(highlight_position = xy)
                    self._icons_shown = True
            else:
                self.hide_icons()	# Untouch
                self._icons_shown = False
                
        pressed = (t == openvr.VREvent_ButtonPress)
        released = (t == openvr.VREvent_ButtonUnpress)
        if (pressed or released) and e.trackedDeviceIndex == self.device_index:
            b = e.data.controller.button
            if b == openvr.k_EButton_SteamVR_Trigger:
                if pressed or self._ui_pressed:
                    ui_click = camera.user_interface.click(pressed, self.room_position.origin())
                    self._ui_pressed = ui_click and pressed
                else:
                    ui_click = None
                from chimerax.core.ui import MouseMode
                if isinstance(ui_click, MouseMode):
                    mmode = ui_click
                    if mmode.name == 'zoom':
                        self._mouse_mode = None
                        self._mode = mname = 'zoom'
                    elif mmode.name in ('rotate', 'translate'):
                        self._mouse_mode = None
                        self._mode = mname = 'move scene'
                    elif hasattr(mmode, 'laser_click') or hasattr(mmode, 'drag_3d'):
                        self._mouse_mode = mmode
                        mmode.enable()
                        self._mode = 'mouse mode'
                        mname = mmode.name
                    msg = 'VR mode %s' % mname
                    self.session.logger.status(msg, log = True)
                elif not ui_click:
                    self._drag = pressed
                    m = self._mode
                    if m == 'mouse mode':
                        self._process_click(camera, pressed)
                    elif m == 'move atoms' and pressed:
                        self.select_sidechain()
                    elif m == 'zoom' and pressed:
                        self._zoom_center = self._pose.origin()
            elif b == openvr.k_EButton_SteamVR_Touchpad:
                if self._icons_shown:
                    if pressed:
                        self.icon_clicked()
            elif b == openvr.k_EButton_ApplicationMenu:
                camera.user_interface.display_ui(pressed, self.room_position)
                self._app_button_down = pressed
            elif b == openvr.k_EButton_Grip:
                if pressed:
                    camera.fit_scene_to_room()

    def process_touchpad_motion(self):
        # Motion on touchpad does not generate an event.
        if self._icons_shown:
            xy = self.touchpad_position()
            if xy is not None:
                self.show_icons(highlight_position = xy)
                    
    def _process_click(self, camera, pressed):
        m = self._mouse_mode
        if hasattr(m, 'laser_click'):
            if pressed:
                p = self.position
                xyz1 = p * (0,0,0)
                range_scene = self._laser_range / camera.scene_scale
                xyz2 = p * (0,0,-range_scene)
                m.laser_click(xyz1, xyz2)
        if hasattr(m, 'drag_3d'):
            if pressed:
                self._last_drag_room_position = self.room_position
            else:
                m.drag_3d(None, None, None)
                self._last_drag_room_position = None
                    
    def _process_drag(self, camera):
        m = self._mouse_mode
        if hasattr(m, 'drag_3d'):
            rp = self.room_position
            ldp = self._last_drag_room_position
            room_move = rp * ldp.inverse()
            delta_z = (rp.origin() - ldp.origin())[1] # Room vertical motion
            rts = camera.room_to_scene
            move = rts * room_move * rts.inverse()
            p = rts * rp
            if m.drag_3d(p, move, delta_z) != 'accumulate drag':
                self._last_drag_room_position = rp
        
    def process_motion(self, camera):
        # For controllers with trigger pressed, use controller motion to move scene
        # Rotation and scaling is about controller position -- has natural feel,
        # like you grab the models where your hand is located.
        # Another idea is to instead pretend controller is at center of models.
        previous_pose = self._pose
        self._update_position(camera)

        if self._app_button_down:
            camera.user_interface.move_ui(self.room_position)

        if not self._drag:
            return

        if previous_pose is None:
            return

        m = self._mode
        pose = self._pose
        if m == 'move scene':
            oc = camera.other_controller(self)
            if oc and oc._drag and oc._mode == 'move scene':
                # Both controllers trying to move scene -- zoom
                self.pinch_zoom(camera, previous_pose.origin(), pose.origin(), oc._pose.origin())
            else:
                move = previous_pose * pose.inverse()
                camera.move_scene(move)
                self._update_position(camera)
        elif m == 'zoom' and self._zoom_center is not None:
            center = self._zoom_center
            move = previous_pose * pose.inverse()
            y_motion = move.matrix[1,3]  # meters
            from math import exp
            s = exp(2*y_motion)
            s = max(min(s, 10.0), 0.1)	# Limit scaling
            from chimerax.core.geometry import distance, translation, scale
            scale = translation(center) * scale(s) * translation(-center)
            camera.move_scene(scale)
            self._update_position(camera)
        elif m == 'move atoms':
            move = pose * previous_pose.inverse()  # Room to room coords
            rts = camera.room_to_scene
            smove = rts * move * rts.inverse()	# Scene to scene coords.
            from chimerax.core.atomic import selected_atoms
            atoms = selected_atoms(self.session)
            atoms.scene_coords = smove * atoms.scene_coords
        elif m == 'mouse mode':
            self._process_drag(camera)
            
    def pinch_zoom(self, camera, prev_pos, pos, other_pos):
        # Two controllers have trigger pressed, scale scene.
        from chimerax.core.geometry import distance, translation, scale
        d, dp = distance(pos,other_pos), distance(prev_pos,other_pos)
        center = 0.5*(pos+other_pos)
        if d > 0.5*dp and dp > 0:
            s = dp / d
            s = max(min(s, 10.0), 0.1)	# Limit scaling
            scale = translation(center) * scale(s) * translation(-center)
            camera.move_scene(scale)
            self._update_position(camera)

    def icon_clicked(self):
        xy = self.touchpad_position()
        if xy is None:
            return
        self.run_icon(xy)

    def select_sidechain(self):
        a = self.closest_atom()
        if a:
            # Residue atoms not including backbone.
            ratoms = a.residue.atoms
            from numpy import logical_not
            scatoms = ratoms.filter(logical_not(ratoms.is_backbones()))
            scatoms.selected = True
        else:
            self.session.selection.clear()

    def select_object_old(self):
        a = self.closest_atom()
        if a:
            # Select atom with bottom of touchpad,
            # or residue with top of touchpad
            xy = self.touchpad_position()
            if xy is not None:
                self.session.selection.clear()
                x,y = xy
                if x >= .5:
                    # Residue atoms not including backbone.
                    ratoms = a.residue.atoms
                    from numpy import logical_not
                    scatoms = ratoms.filter(logical_not(ratoms.is_backbones()))
                    scatoms.selected = True
                if y <= 0:
                    a.selected = True
                else:
                    a.residue.atoms.selected = True

    def touchpad_position(self):
        vrs = self.vr_system
        from ctypes import sizeof
        # TODO: I think pyopenvr eliminated the size arg in Feb 2017.
        import openvr
        size = sizeof(openvr.VRControllerState_t)
        success, cs = vrs.getControllerState(self.device_index, size)
        if success:
            a = cs.rAxis[0]
            return (a.x, a.y)
        return None
    
    def select_atom(self, range = 5.0):
        a = self.closest_atom(range)
        self.session.selection.clear()
        if a is not None:
            a.selected = True
        return a

    def closest_atom(self, range = 5.0):
        atoms = self.displayed_atoms()
        if len(atoms) == 0:
            return None
        xyz = atoms.scene_coords
        tp = self.tip_position()
        d = xyz - tp
        d2 = (d*d).sum(axis = 1)
        i = d2.argmin()
        self.session.selection.clear()
        #print ('closest atom range', d2[i], i, atoms[i], tp, xyz[i], len(atoms))
        if d2[i] > range*range:
            return None
        a = atoms[i]
        return a
        
    def displayed_atoms(self):
        from chimerax.core.atomic import Structure, concatenate, Atoms
        mlist = self.session.models.list(type = Structure)
        matoms = []
        for m in mlist:
            if m.display and m.parents_displayed:
                ma = m.atoms
                matoms.append(ma.filter(ma.displays | (ma.hides != 0)))
        atoms = concatenate(matoms, Atoms)
        return atoms

    def show_icons(self, highlight_position = None):
        d = self.icon_drawing()
        d.display = True
        if highlight_position:
            # x,y ranging from -1 to 1
            x,y = self.icons_position(highlight_position)
            ihd = self.icon_highlight_drawing()
            s = self._cone_length
            aspect = self._icon_rows / self._icon_columns
            from chimerax.core.geometry import translation
            ihd.position = translation((s*x,0,-s*(y+1)*aspect))
            ihd.display = True

    def icons_position(self, xy):
        # Hard to reach corners on circular pad.
        # So expand circle a bit.
        f = 1.2
        x,y = [min(1.0, max(-1.0, f*v)) for v in xy]
        return x,y

    def run_icon(self, xy):
        rows, cols = self._icon_rows, self._icon_columns
        x,y = self.icons_position(xy)
        c = int(0.5 * (x + 1) * cols)
        r = int(0.5 * (1 - y) * rows)
        i = r*cols + c
        s = self._icon_shortcuts
        if i < len(s):
            k = s[i]
            if isinstance(k, str):
                from chimerax.core.commands import run
                run(self.session, 'ks %s' % k)
            else:
                k()	# Python function.  For example, set mouse mode.
                
    def icon_drawing(self):
        d = self._icon_drawing
        if d:
            return d
        
        from chimerax.core.graphics import Drawing
        from chimerax.core.graphics.drawing import rgba_drawing
        self._icon_drawing = d = Drawing('VR icons')
        d.casts_shadows = False
        rgba = self.tiled_icons() # numpy uint8 (ny,nx,4) array
        s = self._cone_length
        h,w = rgba.shape[:2]
        self._icons_aspect = aspect = h/w if w > 0 else 1
        pos = (-s, 0)	# Cone tip is at 0,0
        size = (2*s, 2*s*aspect)
        rgba_drawing(d, rgba, pos, size)
        from chimerax.core.geometry import rotation
        d.vertices = rotation(axis = (1,0,0), angle = -90) * d.vertices
        self.add_drawing(d)
        return d

    def icon_highlight_drawing(self):
        d = self._icon_highlight_drawing
        if d:
            return d
        
        from chimerax.core.graphics import Drawing
        self._icon_highlight_drawing = d = Drawing('VR icon highlight')
        d.casts_shadows = False
        s = self._cone_length
        from chimerax.core.surface import sphere_geometry
        va, na, ta = sphere_geometry(200)
        va *= 0.1*s
        d.geometry = va,ta
        d.normals = na
        d.color = (0,255,0,255)
        self.add_drawing(d)
        return d
        
    def hide_icons(self):
        d = self._icon_drawing
        if d:
            d.display = False
        dh = self._icon_highlight_drawing
        if dh:
            dh.display = False

    def icons(self):
        images = []
        self._icon_shortcuts = ks = []

        from PyQt5.QtGui import QImage

        from os.path import join, dirname
        mm_icon_dir = join(dirname(__file__), '..', 'mouse_modes', 'icons')
        mm = self.session.ui.mouse_modes
        mt = {'translate':self.move_scene,
              'zoom':self.zoom,
              'translate selected models':self.select_and_move}
        modes = [m for m in mm.modes if m.icon_file and m.name in mt]
        for mode in modes:
            icon_path = join(mm_icon_dir, mode.icon_file)
            images.append(QImage(icon_path))
            ks.append(mt[mode.name])
        
        icon_dir = join(dirname(__file__), '..', 'shortcuts', 'icons')
        from ..shortcuts.tool import MoleculeDisplayPanel
        for keys, filename, descrip in MoleculeDisplayPanel.shortcuts:
            icon_path = join(icon_dir, filename)
            images.append(QImage(icon_path))
            ks.append(keys)
        return images

    def tiled_icons(self):
        images = self.icons()
        n = len(images)
        from math import ceil, sqrt
        cols = int(ceil(sqrt(n)))
        rows = (n + cols-1) // cols
        self._icon_columns, self._icon_rows = cols, rows
        isize = self._icon_size
        from PyQt5.QtGui import QImage, QPainter
        from PyQt5.QtCore import QRect, Qt
        ti = QImage(cols*isize, rows*isize, QImage.Format_ARGB32)
        p = QPainter()
        p.begin(ti)
        # Set background white for transparent icons
        p.fillRect(QRect(0,0,cols*isize, rows*isize), Qt.white)
        for i,im in enumerate(images):
            r = QRect((i%cols)*isize, (i//cols)*isize, isize, isize)
            p.drawImage(r, im)
        from chimerax.core.graphics import qimage_to_numpy
        rgba = qimage_to_numpy(ti)
        p.end()
        return rgba

    def move_scene(self):
        self._mode = 'move scene'
    def zoom(self):
        self._mode = 'zoom'
    def select_and_move(self):
        self._mode = 'move atoms'
            
def hmd44_to_opengl44(hm44):
    from numpy import array, float32
    m = hm44.m
    m44 = array(((m[0][0], m[1][0], m[2][0], m[3][0]),
                 (m[0][1], m[1][1], m[2][1], m[3][1]), 
                 (m[0][2], m[1][2], m[2][2], m[3][2]), 
                 (m[0][3], m[1][3], m[2][3], m[3][3]),),
                float32)
    return m44

def hmd34_to_position(hmat34):
    from chimerax.core.geometry import Place
    from numpy import array, float32
    p = Place(array(hmat34.m, float32))
    return p
    
