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
def vr(session, enable = None, room_position = None, mirror = True, desktop_view = True,
       show_controllers = True, multishadow_allowed = False, simplify_graphics = True,
       toolbar_panels = True, icons = False):
    '''
    Enable stereo viewing and head motion tracking with virtual reality headsets using SteamVR.

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
      Whether to display the VR user's view point or a fixed view point in the room in the
      desktop graphics window.
    desktop_view : bool
      Whether to continue rendering the desktop graphics window.  This also turns off waiting
      for display vertical sync on the computer monitor so that the 60 Hz refresh rate
      does not slow down the 90 Hz rendering to the VR headset.
    show_controllers : bool
      Whether to show the hand controllers in the scene. Default true.
    multishadow_allowed : bool
      If this option is false and multi-shadow lighting is enabled (ambient occlusion) when vr is
      enabled, then lighting is switched to simple lighting.  If the option is true then no
      changes to lighting mode are made.  Often rendering is not fast enough
      to support multishadow lighting so this option makes sure it is off so that stuttering
      does not occur.  Default False.
    simplify_graphics : bool
      Adjust level-of-detail total number of triangles for atoms and bonds to a reduced value
      when VR is enabled, and restore to default value when VR disabled.  This helps maintain
      full rendering speed in VR.  Default true.
    toolbar_panels : bool
      Whether to hide mouse modes and shortcut toolbars and instead show them as tool panels.
      This is useful for consolidating the controls in the VR gui panel.  Default true.
    icons : bool
      Experimental.  Superceded by embedded gui panel.
      Whether to show a panel of icons when controller trackpad is touched.
      For demonstrations the icons can be too complex and it is better not to have icons.
      Default false.
    '''
    
    if enable is None and room_position is None:
        enable = True

    if enable is not None:
        if enable:
            start_vr(session, multishadow_allowed, simplify_graphics,
                     main_camera = mirror or not desktop_view)
        else:
            stop_vr(session, simplify_graphics)

    c = vr_camera(session)
    if room_position is not None:
        if c is None:
            from chimerax.core.errors import UserError
            raise UserError('Cannot use vr roomPosition unless vr enabled.')
        if isinstance(room_position, str) and room_position == 'report':
            p = ','.join('%.5g' % x for x in tuple(c.room_to_scene.matrix.flat))
            session.logger.info(p)
        else:
            c.room_to_scene = room_position
            c._last_position = c.position

    if c:
        if desktop_view is not None:
            wait_for_vsync(session, desktop_view)
        if mirror is not None:
            c.mirror_display = mirror
        if show_controllers is not None:
            for hc in c.hand_controllers(show_controllers):
                hc.show_in_scene(show_controllers)
        if icons is not None: 
            for hc in c.hand_controllers():
                hc.enable_icon_panel(icons)

    if toolbar_panels:
        from chimerax.mouse_modes.tool import MouseModePanel
        from chimerax.shortcuts.tool import ShortcutPanel
        toolbar_classes = (MouseModePanel, ShortcutPanel)
        for tb in session.tools.list():
            if isinstance(tb, toolbar_classes):
                tb.display(False)
                tb.display_panel(True)

# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, NoArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('mirror', BoolArg),
                              ('desktop_view', BoolArg),
                              ('show_controllers', BoolArg),
                              ('multishadow_allowed', BoolArg),
                              ('simplify_graphics', BoolArg),
                              ('toolbar_panels', BoolArg),
                              ('icons', BoolArg),
                   ],
                   synopsis = 'Start SteamVR virtual reality rendering')
    register('device vr', desc, vr, logger=logger)
    create_alias('vr', 'device vr $*', logger=logger)

# -----------------------------------------------------------------------------
#
def start_vr(session, multishadow_allowed = False, simplify_graphics = True, main_camera = True):

    v = session.main_view
    if not multishadow_allowed and v.lighting.multishadow > 0:
        from chimerax.core.commands import run
        run(session, 'lighting simple')

    if simplify_graphics:
        from chimerax.std_commands.graphics import graphics
        graphics(session, total_atom_triangles=1000000, total_bond_triangles=1000000)

    # TODO: Handle switching vr to/from main_camera when vr already started.
    
    if vr_camera(session) is not None:
        return

    try:
        import openvr
    except Exception as e:
        from chimerax.core.errors import UserError
        raise UserError('Failed to import OpenVR module: %s' % str(e))

    _create_vr_view(session, main_camera)
    
    # Set redraw timer to redraw as soon as Qt events processsed to minimize dropped frames.
    session.update_loop.set_redraw_interval(0)

    msg = 'started SteamVR rendering'
    log = session.logger
    log.status(msg)
    log.info(msg)

# -----------------------------------------------------------------------------
# If desktop view is independent of vr view (main_camera = False) then create a
# separate View instance for VR.
#
def _create_vr_view(session, main_camera):
    mv = session.main_view
    if main_camera:
        vrv = mv
    else:
        from chimerax.core.graphics import View
        vrv = View(mv.drawing)
        vrv.lighting = mv.lighting
        vrv.material = mv.material
        vrv.initialize_rendering(mv.render.opengl_context)
    session._vr_view = vrv
    vrv.camera = SteamVRCamera(session)
    if not main_camera:
        h = session.triggers.add_handler('frame drawn', lambda *unused: vrv.draw(check_for_changes = False))
        vrv._draw_handler = h
        
# -----------------------------------------------------------------------------
#
def _remove_vr_view(session):
    vrv = getattr(session, '_vr_view', None)
    if vrv and vrv is not session.main_view:
        delattr(session, '_vr_view')
        session.triggers.remove_handler(vrv._draw_handler)
        vrv.delete()
        
# -----------------------------------------------------------------------------
#
def vr_camera(session):
    vrv = getattr(session, '_vr_view', None)
    if vrv:
        c = vrv.camera
        if isinstance(c, SteamVRCamera):
            return c
    return None

# -----------------------------------------------------------------------------
#
def stop_vr(session, simplify_graphics = True):

    c = vr_camera(session)
    if c is None:
        return
    
    # Have to delay shutdown of SteamVR connection until draw callback
    # otherwise it clobbers the Qt OpenGL context making entire gui black.
    def replace_camera(s = session):
        from chimerax.core.graphics import MonoCamera
        v = s.main_view
        v.camera = MonoCamera()
        s.update_loop.set_redraw_interval(10)
        if simplify_graphics:
            from chimerax.std_commands.graphics import graphics
            graphics(session, total_atom_triangles=5000000, total_bond_triangles=5000000)
        v.view_all()

    _remove_vr_view(session)
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

        from chimerax.core.geometry import Place
        self.room_position = Place()	# Camera position in room coordinates
        self._room_to_scene = None	# Maps room coordinates to scene coordinates

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
        self._frame_started = False
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

    def _get_room_to_scene(self):
        return self._room_to_scene
    def _set_room_to_scene(self, p):
        self._update_desktop_camera(p)
        self._room_to_scene = p
        self._reposition_user_interface()
    room_to_scene = property(_get_room_to_scene, _set_room_to_scene)

    def _update_desktop_camera(self, new_rts):
        mc = self._session.main_view.camera
        if self is mc:
            return  # VR and desktop cameras are the same

        # Main camera stays at same position in room.
        rts = self._room_to_scene
        if rts is None:
            return   # VR room to scene not yet set.  Leave desktop camera unchanged.
        tf = new_rts * rts.inverse()
        mpos = tf * mc.position
        # Need to remove scale factor.
        x,y,z = tf.matrix[:,0]
        from math import sqrt
        s = 1/sqrt(x*x + y*y + z*z)
        mpos.matrix[:3,:3] *= s
        mc.position = mpos
    
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
            b = self.vr_view.drawing_bounds()
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
        
    def _reposition_user_interface(self):
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
            self.render.make_current()
            fb.delete()
            self._framebuffer = None
        if self._close_cb:
            self._close_cb()	# Replaces the main view camera and resets redraw rate.

        
    def name(self):
        '''Name of camera.'''
        return 'vr'

    @property
    def vr_view(self):
        return self._session._vr_view

    @property
    def render(self):
        return self._session._vr_view.render
    
    def _start_frame(self):
        import openvr
        c = self.compositor
        if c is None or self._close:
            return
        c.waitGetPoses(self._poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
        self._frame_started = True

    def next_frame(self, *_):
        c = self.compositor
        if c is None or self._close:
            return
        self._start_frame()
        import openvr
        hmd_pose0 = self._poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not hmd_pose0.bPoseIsValid:
            return
        # head to room coordinates.
        H = hmd34_to_position(hmd_pose0.mDeviceToAbsoluteTracking)

        self.process_controller_events()

        self.user_interface.update_if_needed()
        
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
            if hc.uses_touch_motion():
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
        if not self._frame_started:
            self._start_frame()	# Window resize causes draw without new frame trigger.
        fb = self._texture_framebuffer(render)
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

        self._frame_started = False

    def _texture_framebuffer(self, render):

        tw,th = self._render_size
        fb = self._framebuffer
        if fb is None or fb.width != tw or fb.height != th:
            from chimerax.core.graphics import Texture, opengl
            t = Texture()
            t.initialize_rgba((tw,th))
            self._framebuffer = fb = opengl.Framebuffer('VR', render.opengl_context, color_texture = t)
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
        self._width = 0.5		# Billboard width in room coords, meters.
        self._height = None		# Height in room coords determined by window aspect and width.
        self._panel_size = None 	# Panel size in pixels
        self._panel_offset = (0,0)  	# Offset from desktop main window upper left corner, to panel rectangle 
        self._ui_click_range = 0.05 	# Maximum distance of click from plane, room coords, meters.
        self._update_later = 0		# Redraw panel after this many frames
        self._update_delay = 10		# After click on panel, update after this number of frames
        self._ui_drawing = None
        self._start_ui_move_time = None
        self._last_ui_position = None
        self._ui_hide_time = 0.3	# seconds. Max application button press/release time to hide ui
        self.button_down = None

        # Buttons that can be pressed on user interface.
        import openvr
        self.buttons = (openvr.k_EButton_SteamVR_Trigger, openvr.k_EButton_Grip, openvr.k_EButton_SteamVR_Touchpad)
        
    def close(self):
        ui = self._ui_drawing
        if ui:
            self._session.models.close([ui])
            self._ui_drawing = None

    def shown(self):
        ui = self._ui_drawing
        if ui is None:
            return False
        if ui.deleted:
            self._ui_drawing = None
            return False
        return ui.display
    
    def show(self, room_position):
        ui = self._ui_drawing
        if ui is None:
            self._ui_drawing = ui = self._create_ui_drawing()
        self._update_ui_image()
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

    def click_position(self, room_point):
        if not self.shown():
            return None, False
        ui = self._ui_drawing
        x,y,z = ui.room_position.inverse() * room_point
        hw, hh = 0.5*self._width, 0.5*self._height
        cr = self._ui_click_range
        on_panel = (x >= -hw and x <= hw and y >= -hh and y <= hh and z >= -cr and z <= cr)
        sx, sy = self._panel_size
        ox, oy = self._panel_offset
        px, py = ox + sx * (x + hw) / (2*hw), oy + sy * (hh - y) / (2*hh)
        return (px,py), on_panel

    def press(self, window_xy):
        return self._click('press', window_xy)

    def drag(self, window_xy):
        return self._click('move', window_xy)

    def release(self, window_xy):
        return self._click('release', window_xy)

    def _click(self, type, window_xy):
        '''Type can be "press" or "release".'''
        if self._post_mouse_event(type, window_xy) and type != 'move':
            self.redraw_ui()
            return True
        return False

    def redraw_ui(self):
        self._update_later = self._update_delay
        self._update_ui_image()

    def update_if_needed(self):
        if self.shown() and self._update_later:
            self._update_later -= 1
            if self._update_later == 0:
                self._update_ui_image()
                
    def clicked_mouse_mode(self, window_xy):
        w, pos = self._clicked_widget(window_xy)
        from PyQt5.QtWidgets import QToolButton
        if isinstance(w, QToolButton):
            a = w.defaultAction()
            if hasattr(a, 'mouse_mode'):
                return self._hand_mode(a.mouse_mode)
        return None

    def _hand_mode(self, mouse_mode):
        name = mouse_mode.name
        if name == 'zoom':
            m = ZoomMode()
        elif name in ('rotate', 'translate'):
            m = MoveSceneMode()
        else:
            m = MouseMode(mouse_mode)
        return m
    
    def _post_mouse_event(self, type, window_xy):
        '''Type is "press", "release" or "move".'''
        w, pos = self._clicked_widget(window_xy)
        if w is None or pos is None:
            return False
        from PyQt5.QtGui import QMouseEvent
        from PyQt5.QtCore import Qt, QEvent
        if type == 'press':
            et = QEvent.MouseButtonPress
            button = buttons = Qt.LeftButton
        elif type == 'release':
            et = QEvent.MouseButtonRelease
            button = Qt.LeftButton
            buttons =  Qt.NoButton
        elif type == 'move':
            et = QEvent.MouseMove
            button =  Qt.NoButton
            buttons = Qt.LeftButton
        me = QMouseEvent(et, pos, button, buttons, Qt.NoModifier)
        self._session.ui.postEvent(w, me)
        return True
        
    def _clicked_widget(self, window_xy):
        ui = self._session.ui
        mw = ui.main_window
        from PyQt5.QtCore import QPoint, QPointF
        x,y = window_xy
        gp = mw.mapToGlobal(QPoint(int(x), int(y)))
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

    def _update_ui_image(self):
        rgba = self._panel_image()
        h,w = rgba.shape[:2]
        aspect = h/w
        rw = self._width		# Billboard width in room coordinates
        self._height = rh = aspect * rw
        self._session._vr_view.render.make_current()	# Required OpenGL context for replacing texture.
        from chimerax.core.graphics.drawing import rgba_drawing
        rgba_drawing(self._ui_drawing, rgba, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))

    def _panel_image(self):
        ui = self._session.ui
        im = ui.window_image()
        from chimerax.core.graphics.drawing import qimage_to_numpy
        rgba = qimage_to_numpy(im)
        gw = ui.main_window.graphics_window
        self._panel_offset = (ox, oy) = (gw.x() + gw.width(), gw.y())
        ph = gw.height()
        wh,ww = rgba.shape[:2]
        prgba = rgba[wh-(ph+oy):wh-oy,ox:,:]
        h,w = prgba.shape[:2]
        self._panel_size = (w, h)
        return prgba

    def display_ui(self, button_pressed, hand_room_position, camera_position):
        if button_pressed:
            rp = hand_room_position
            self._last_ui_position = rp
            if self.shown():
                from time import time
                self._start_ui_move_time = time()
            else:
                # Orient horizontally and perpendicular to floor
                view_axis = camera_position.origin() - rp.origin()
                from chimerax.core.geometry import orthonormal_frame, translation
                p = orthonormal_frame(view_axis, (0,1,0), origin = rp.origin())
                p = translation(0.5 * self._width * p.axes()[1]) * p
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

    def scale_ui(self, scale_factor):
        self._width *= scale_factor
        self._height *= scale_factor
        rw, rh = self._width, self._height
        from chimerax.core.graphics.drawing import resize_rgba_drawing
        resize_rgba_drawing(self._ui_drawing, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))
        
from chimerax.core.models import Model
class HandControllerModel(Model):
    casts_shadows = False
    pickable = False
    skip_bounds = True
    _controller_colors = ((200,200,0,255), (0,200,200,255))
    SESSION_SAVE = False

    def __init__(self, device_index, session, vr_system, show = True, size = 0.20, aspect = 0.2):
        name = 'Hand %s' % device_index
        Model.__init__(self, name, session)
        self.device_index = device_index
        self.vr_system = vr_system

        self._pose = None
        self._previous_pose = None
        self.room_position = None	# Hand controller position in room coordinates.

        # Assign actions bound to controller buttons
        import openvr
        self._modes = {			# Maps button name to HandMode
            openvr.k_EButton_SteamVR_Trigger: MoveSceneMode(),
            openvr.k_EButton_Grip: RecenterMode(),
            openvr.k_EButton_ApplicationMenu: ShowUIMode(),
        }
        self._active_drag_modes = set() # Modes with an active drag (ie. button down and not yet released).

        # Draw controller as a cone.
        self._create_model_geometry(size, aspect)

        self._shown_in_scene = show
        if show:
            session.models.add([self])

    def _create_model_geometry(self, size, aspect):
        from chimerax.surface.shapes import cone_geometry
        va, na, ta = cone_geometry(nc = 50, points_up = False)
        va[:,:2] *= aspect
        va[:,2] += 0.5		# Move tip to 0,0,0 for picking
        va *= size
        cc = self._controller_colors
        from numpy import array, uint8
        rgba8 = array(cc[self.device_index%len(cc)], uint8)

        self._cone_vertices = va
        self._last_cone_scale = 1
        self._cone_length = size
        self.set_geometry(va, na, ta)
        self.color = array(rgba8, uint8)
            
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
                va = (1/s) * self._cone_vertices
                self.set_geometry(va, self.normals, self.triangles)
                self._last_cone_scale = s

    def tip_position(self):
        return self.scene_position.origin()

    def enable_icon_panel(self, enable):
        from openvr import k_EButton_SteamVR_Touchpad as touchpad
        if enable:
            self._modes[touchpad] = IconPanel()
        elif touchpad in self._modes:
            del self._modes[touchpad]
            
    def process_event(self, e, camera):

        if e.trackedDeviceIndex != self.device_index:
            return

        # Handle trackpad touch events.  This is diffent from a button press.
        if self._process_touch_event(e):
            return

        # Handle button press events.
        t = e.eventType
        import openvr
        pressed = (t == openvr.VREvent_ButtonPress)
        released = (t == openvr.VREvent_ButtonUnpress)
        if not pressed and not released:
            return

        # Check for click on user interface panel.
        b = e.data.controller.button
        if self._process_ui_event(camera.user_interface, b, pressed, released):
            return
        
        # Call HandMode press() or release() callback.
        m = self._modes.get(b)
        if m:
            adm = self._active_drag_modes
            if pressed:
                m.pressed(camera, self)
                adm.add(m)
            elif m in adm:
                m.released(camera, self)
                adm.remove(m)

    def _process_ui_event(self, ui, b, pressed, released):
        if b not in ui.buttons:
            return False
        
        window_xy, on_panel = ui.click_position(self.room_position.origin())
        if released and ui.button_down == (self, b) and window_xy:
            # Always release mouse button even if off panel.
            ui.release(window_xy)
            ui.button_down = None
        elif on_panel:
            if pressed and ui.button_down is None:
                hand_mode = ui.clicked_mouse_mode(window_xy)
                if hand_mode:
                    if isinstance(hand_mode, MouseMode) and not hand_mode.has_vr_support:
                        msg = 'No VR support for mouse mode %s' % hand_mode.name
                    else:
                        self._modes[b] = hand_mode
                        msg = 'VR mode %s' % hand_mode.name
                    self.session.logger.status(msg, log = True)
                    ui.redraw_ui()	# Show log message
                else:
                    ui.press(window_xy)
                    ui.button_down = (self, b)
            elif released and ui.button_down == (self, b):
                ui.release(window_xy)
                ui.button_down = None
        else:
            return False
        return True

    def _process_touch_event(self, e):
        t = e.eventType
        import openvr
        if ((t == openvr.VREvent_ButtonTouch or t == openvr.VREvent_ButtonUntouch)
            and e.data.controller.button == openvr.k_EButton_SteamVR_Touchpad):
            m = self._modes.get(openvr.k_EButton_SteamVR_Touchpad)
            if m:
                if t == openvr.VREvent_ButtonTouch:
                    m.touch()
                else:
                    m.untouch()
            return True
        return False

    def uses_touch_motion(self):
        import openvr
        m = self._modes.get(openvr.k_EButton_SteamVR_Touchpad)
        return m.uses_touch_motion if m else False
        
    def process_touchpad_motion(self):
        # Motion on touchpad does not generate an event.
        if self._icons_shown:
            xy = self.touchpad_position()
            if xy is not None:
                self.show_icons(highlight_position = xy)
        
    def process_motion(self, camera):
        # Move hand controller model
        previous_pose = self._pose
        self._update_position(camera)

        # Generate mouse move event on ui panel.
        ui = camera.user_interface
        if ui.button_down and ui.button_down[0] == self:
            window_xy, on_panel = ui.click_position(self.room_position.origin())
            ui.drag(window_xy)
            return

        # Do hand controller drag when buttons pressed
        if previous_pose is not None:
            pose = self._pose
            for m in self._active_drag_modes:
                m.drag(camera, self, previous_pose, pose)

class HandMode:
    def pressed(self, camera, hand_controller):
        pass
    def released(self, camera, hand_controller):
        pass
    def drag(self, camera, hand_controller, previous_pose, pose):
        pass
    uses_touch_motion = False
    def touch(self):
        pass
    def untouch(self):
        pass

class ShowUIMode(HandMode):
    def pressed(self, camera, hand_controller):
        camera.user_interface.display_ui(True, hand_controller.room_position, camera.room_position)
    def released(self, camera, hand_controller):
        camera.user_interface.display_ui(False, hand_controller.room_position, camera.room_position)
    def drag(self, camera, hand_controller, previous_pose, pose):
        oc = camera.other_controller(hand_controller)
        if oc and self._ui_zoom(oc):
            scale, center = _pinch_scale(previous_pose.origin(), pose.origin(), oc._pose.origin())
            if scale is not None:
                camera.user_interface.scale_ui(scale)
        else:
            camera.user_interface.move_ui(hand_controller.room_position)
    def _ui_zoom(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, ShowUIMode):
                return True
        return False

class MoveSceneMode(HandMode):
    name = 'move scene'
    def drag(self, camera, hand_controller, previous_pose, pose):
        oc = camera.other_controller(hand_controller)
        if oc and self._other_controller_move(oc):
            # Both controllers trying to move scene -- zoom
            scale, center = _pinch_scale(previous_pose.origin(), pose.origin(), oc._pose.origin())
            if scale is not None:
                self._pinch_zoom(camera, hand_controller, center, scale)
        else:
            move = previous_pose * pose.inverse()
            camera.move_scene(move)
            hand_controller._update_position(camera)
    def _other_controller_move(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, MoveSceneMode):
                return True
        return False
    def _pinch_zoom(self, camera, hand_controller, center, scale_factor):
        # Two controllers have trigger pressed, scale scene.
        from chimerax.core.geometry import translation, scale
        scale = translation(center) * scale(1/scale_factor) * translation(-center)
        camera.move_scene(scale)
        hand_controller._update_position(camera)

def _pinch_scale(prev_pos, pos, other_pos):
    from chimerax.core.geometry import distance
    d, dp = distance(pos,other_pos), distance(prev_pos,other_pos)
    if dp > 0:
        s = d / dp
        s = max(min(s, 10.0), 0.1)	# Limit scaling
        center = 0.5*(pos+other_pos)
        return s, center
    return None, None

class ZoomMode(HandMode):
    name = 'zoom'
    def __init__(self):
        self._zoom_center = None
    def pressed(self, camera, hand_controller):
        self._zoom_center = hand_controller._pose.origin()
    def drag(self, camera, hand_controller, previous_pose, pose):
        if self._zoom_center is None:
            return
        center = self._zoom_center
        move = previous_pose * pose.inverse()
        y_motion = move.matrix[1,3]  # meters
        from math import exp
        s = exp(2*y_motion)
        s = max(min(s, 10.0), 0.1)	# Limit scaling
        from chimerax.core.geometry import distance, translation, scale
        scale = translation(center) * scale(s) * translation(-center)
        camera.move_scene(scale)
        hand_controller._update_position(camera)

class RecenterMode(HandMode):
    name = 'recenter'
    def pressed(self, camera, hand_controller):
        camera.fit_scene_to_room()

class MouseMode(HandMode):
    name = 'mouse mode'
    def __init__(self, mouse_mode):
        self._mouse_mode = mouse_mode
        mouse_mode.enable()
        self.name = mouse_mode.name
        self._last_drag_room_position = None # Hand controller position at last drag_3d call
        self._laser_range = 5		# Range for mouse mode laser clicks

    @property
    def has_vr_support(self):
        m = self._mouse_mode
        return hasattr(m, 'laser_click') or hasattr(m, 'drag_3d')
    
    def pressed(self, camera, hand_controller):
        self._click(camera, hand_controller, True)

    def released(self, camera, hand_controller):
        self._click(camera, hand_controller, False)

    def _click(self, camera, hand_controller, pressed):
        m = self._mouse_mode
        if hasattr(m, 'laser_click'):
            if pressed:
                p = hand_controller.position
                xyz1 = p * (0,0,0)
                range_scene = self._laser_range / camera.scene_scale
                xyz2 = p * (0,0,-range_scene)
                m.laser_click(xyz1, xyz2)
        if hasattr(m, 'drag_3d'):
            if pressed:
                self._last_drag_room_position = hand_controller.room_position
            else:
                m.drag_3d(None, None, None)
                self._last_drag_room_position = None

    def drag(self, camera, hand_controller, previous_pose, pose):
        m = self._mouse_mode
        if hasattr(m, 'drag_3d'):
            rp = hand_controller.room_position
            ldp = self._last_drag_room_position
            room_move = rp * ldp.inverse()
            delta_z = (rp.origin() - ldp.origin())[1] # Room vertical motion
            rts = camera.room_to_scene
            move = rts * room_move * rts.inverse()
            p = rts * rp
            if m.drag_3d(p, move, delta_z) != 'accumulate drag':
                self._last_drag_room_position = rp
        
class MoveAtomsMode(HandMode):
    name = 'move atoms'
    def pressed(self, camera, hand_controller):
        # TODO: Need to get icon panel
        ip = hand_controller._icon_panel
        ip.select_sidechain()
    def drag(self, camera, hand_controller, previous_pose, pose):
        move = pose * previous_pose.inverse()  # Room to room coords
        rts = camera.room_to_scene
        smove = rts * move * rts.inverse()	# Scene to scene coords.
        from chimerax.atomic import selected_atoms
        atoms = selected_atoms(camera._session)
        atoms.scene_coords = smove * atoms.scene_coords

class IconPanel(HandMode):
    def __init__(self):
        self._icon_drawing = None
        self._icon_highlight_drawing = None
        self._icon_size = 128  # pixels
        self._icon_columns = 0
        self._icon_rows = 0
        self._icon_shortcuts = []
        self._icons_shown = False

    def touch(self):
        xy = self.touchpad_position()
        if xy is not None:
            self.show_icons(highlight_position = xy)
            self._icons_shown = True

    def untouch(self):
        self.hide_icons()	# Untouch
        self._icons_shown = False

    def pressed(self, camera, hand_controller):
        if self._icons_shown:
            self.icon_clicked()
    
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
        from chimerax.atomic import Structure, concatenate, Atoms
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
        v = rotation(axis = (1,0,0), angle = -90) * d.vertices
        d.set_geometry(v, d.normals, d.triangles)
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
        from chimerax.surface import sphere_geometry
        va, na, ta = sphere_geometry(200)
        va *= 0.1*s
        d.set_geometry(va, na, ta)
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
    
