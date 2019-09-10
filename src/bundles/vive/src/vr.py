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
def vr(session, enable = None, room_position = None, display = None,
       show_controllers = True, gui = None, click_range = None,
       multishadow_allowed = False, simplify_graphics = True,
       toolbar_panels = True):
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
    display : "mirror", "independent", or "blank"
      Controls what is shown on the desktop display.  The default "mirror" shows the right
      eye view seen in the VR headset.  With "independent" the desktop display shows a
      separate camera view fixed in the VR room coordinates set to match the viewpoint of
      the VR headset when the command is issued. The value "blank" displays no graphics on
      the desktop display which allows all graphics computing resources to be dedicated to
      the VR headset rendering.
    show_controllers : bool
      Whether to show the hand controllers in the scene. Default true.
    gui : string
      Name of a tool instance which will be shown as the VR gui panel.  If not specified
      then the VR gui panel consists of all tools docked on the right side of the main window.
    click_range : float
      How far away hand controller tip can be when clicking an atom in scene units
      (Angstroms).  Default 5.
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
    '''
    
    if enable is None and room_position is None:
        enable = True

    start = (vr_camera(session) is None)

    if enable is not None:
        if enable:
            start_vr(session, multishadow_allowed, simplify_graphics)
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

    if c:
        if display is None and start:
            if not wait_for_vsync(session, False):
                session.logger.warning('Graphics on desktop display may cause VR to flicker. Turning off mirroring to desktop display.')
                display = 'blank'
        if display is not None:
            if display in ('mirror', 'independent'):
                if not wait_for_vsync(session, False):
                    session.logger.warning('Graphics on desktop display may cause VR to flicker.')
            c.desktop_display = display
            if display == 'independent':
                c.initialize_desktop_camera_position = True
        if show_controllers is not None:
            c.show_hand_controllers(show_controllers)
        if gui is not None:
            c.user_interface.set_gui_panels([tool_name.strip() for tool_name in gui.split(',')])
        if click_range is not None:
            c.user_interface.set_mouse_mode_click_range(click_range)

    if toolbar_panels:
        from chimerax.mouse_modes.tool import MouseModePanel
        from chimerax.shortcuts.tool import ShortcutPanel
        toolbar_classes = (MouseModePanel, ShortcutPanel)
        for tb in session.tools.list():
            if isinstance(tb, toolbar_classes) and tb.displayed():
                tb.display(False)
                tb.display_panel(True)

# -----------------------------------------------------------------------------
# Assign VR hand controller buttons
#
def vr_button(session, button, mode, hand = None):
    '''
    Assign VR hand controller buttons

    Parameters
    ----------
    button : 'trigger', 'grip', 'touchpad', 'menu', 'A', 'B', 'X', 'Y'
      Name of button to assign.  Buttons A/B are for Oculus controllers and imply hand = 'right',
      and X/Y imply hand = 'left'
    mode : HandMode instance
      VR hand mode to assign to button.
    hand : 'left', 'right', None
      Which hand controller to assign.  If None then assign button on both hand controllers.
      If button is A, B, X, or Y then hand is ignored since A/B implies right and X/Y implies left.
    '''

    c = vr_camera(session)
    if c is None:
        from chimerax.core.errors import UserError
        raise UserError('Must have VR started (command "vr on") before assigning VR buttons')

    if button in ('A', 'B'):
        hand = 'right'
    elif button in ('X', 'Y'):
        hand = 'left'
        
    hclist = [hc for hc in c.hand_controllers() if hand is None or hc.left_or_right == hand]
    if len(hclist) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Hand controller is not enabled.')

    import openvr
    openvr_buttons = {
        'grip': openvr.k_EButton_Grip,
        'menu': openvr.k_EButton_ApplicationMenu,
        'trigger': openvr.k_EButton_SteamVR_Trigger,
        'touchpad': openvr.k_EButton_SteamVR_Touchpad,
        'A': openvr.k_EButton_A,
        'B': openvr.k_EButton_ApplicationMenu,
        'X': openvr.k_EButton_A,
        'Y': openvr.k_EButton_ApplicationMenu
    }
    openvr_button = openvr_buttons[button]

    for hc in hclist:
        hc._set_hand_mode(openvr_button, mode)

# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, StringArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('display', EnumOf(('mirror', 'independent', 'blank'))),
                              ('show_controllers', BoolArg),
                              ('gui', StringArg),
                              ('click_range', FloatArg),
                              ('multishadow_allowed', BoolArg),
                              ('simplify_graphics', BoolArg),
                              ('toolbar_panels', BoolArg),
                   ],
                   synopsis = 'Start SteamVR virtual reality rendering')
    register('device vr', desc, vr, logger=logger)
    create_alias('vr', 'device vr $*', logger=logger)

    desc = CmdDesc(required = [('button', EnumOf(('trigger', 'grip', 'touchpad', 'menu', 'A', 'B', 'X', 'Y'))),
                               ('mode', VRModeArg)],
                   keyword = [('hand', EnumOf(('left', 'right')))],
                   synopsis = 'Assign VR hand controller buttons')
    register('device vr button', desc, vr_button, logger=logger)
    create_alias('vr button', 'device vr button $*', logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError
class VRModeArg(Annotation):
    '''Command argument for specifying VR hand controller mode.'''

    @staticmethod
    def parse(text, session):
        modes = session.ui.mouse_modes.modes
        from chimerax.core.commands import EnumOf
        mode_arg = EnumOf(tuple(m.name for m in modes))
        value, used, rest = mode_arg.parse(text, session)
        mmap = {m.name:m for m in modes}
        mouse_mode = mmap[value]
        c = vr_camera(session)
        if c is None:
            raise AnnotationError('VR hand modes can only be specified when VR is enabled')
        hm = c.user_interface._hand_mode(mouse_mode)
        return hm, used, rest

# -----------------------------------------------------------------------------
#
def start_vr(session, multishadow_allowed = False, simplify_graphics = True, label_reorient = 45):

    v = session.main_view
    if not multishadow_allowed and v.lighting.multishadow > 0:
        from chimerax.core.commands import run
        run(session, 'lighting simple')

    if simplify_graphics:
        from chimerax.std_commands.graphics import graphics_quality
        graphics_quality(session, total_atom_triangles=1000000, total_bond_triangles=1000000)

    from chimerax.label.label3d import label_orient
    label_orient(session, label_reorient)	# Don't continuously reorient labels.
    
    if vr_camera(session) is not None:
        return

    try:
        import openvr
    except Exception as e:
        from chimerax.core.errors import UserError
        raise UserError('Failed to import OpenVR module: %s' % str(e))

    import sys
    if sys.platform == 'darwin':
        # SteamVR on Mac is older then what PyOpenVR expects.
        openvr.IVRSystem_Version = "IVRSystem_019"
        
    mv = session.main_view
    try:
        mv.camera = SteamVRCamera(session)
    except openvr.OpenVRError as e:
        if 'error number 108' in str(e):
            msg = ('The VR headset was not detected.\n' +
                   'Possibly a cable to the VR headset is not plugged in.\n' +
                   'If the headset is a Vive Pro, the link box may be turned off.\n' +
                   'If using a Vive Pro wireless adapter it may not be powered on.')
        else:
            msg = ('Failed to initialize OpenVR.\n' +
                   'Possibly SteamVR is not installed or it failed to start.')
        from chimerax.core.errors import UserError
        raise UserError('%s\n%s' % (msg, str(e)))
        
    
    # Set redraw timer to redraw as soon as Qt events processsed to minimize dropped frames.
    session.update_loop.set_redraw_interval(0)

    msg = 'started SteamVR rendering'
    log = session.logger
    log.status(msg)
    log.info(msg)
        
# -----------------------------------------------------------------------------
#
def vr_camera(session):
    c = session.main_view.camera
    return c if isinstance(c, SteamVRCamera) and not c.closed else None

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
            from chimerax.std_commands.graphics import graphics_quality
            graphics_quality(session, total_atom_triangles=5000000, total_bond_triangles=5000000)
        from chimerax.label.label3d import label_orient
        label_orient(session, 0)	# Continuously reorient labels.
        v.view_all()

    c.close(replace_camera)
    wait_for_vsync(session, True)

# -----------------------------------------------------------------------------
#
def wait_for_vsync(session, wait):
    r = session.main_view.render
    r.make_current()
    return r.wait_for_vsync(wait)

# -----------------------------------------------------------------------------
#
from chimerax.core.graphics import Camera
class SteamVRCamera(Camera):

    always_draw = True	# Draw even if main window iconified.
    
    def __init__(self, session):

        Camera.__init__(self)

        self._session = session
        self._framebuffers = []		# For rendering each eye view to a texture
        self._texture_drawing = None	# For desktop graphics display
        from sys import platform
        self._use_opengl_flush = (platform == 'darwin')	# On macOS 10.14.1 flickers without glFlush().

        from chimerax.core.geometry import Place

        self._close = False
        self._hand_controllers = []	# List of HandController
        self._controller_show = True	# Whether to show hand controllers
        self._controller_next_id = 0	# Used when searching for controllers.
        self.user_interface = UserInterface(self, session)
        self._vr_model_group = None	# Grouping model for hand controllers and UI models
        self._vr_model_group_id = 100	# Keep VR model group at bottom of model panel

        self.desktop_display = 'mirror'	# What to show in desktop graphics window, 'mirror', 'independent' or 'blank'.
        self._desktop_camera_position = Place()	#  Used only for desktop_display = "independent" mode.
        self.desktop_field_of_view = 90		# Degrees. Used only for desktop_display = "independent" mode.
        self.initialize_desktop_camera_position = False

        self.room_position = Place()	# ChimeraX camera coordinates to room coordinates
        self._room_to_scene = None	# Maps room coordinates to scene coordinates

        import openvr
        self.vr_system = vrs = openvr.init(openvr.VRApplication_Scene)
        # The init() call raises OpenVRError if SteamVR is not installed.
        # Handle this in the code that tries to create the camera.

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
        t = session.triggers
        self._new_frame_handler = t.add_handler('new frame', self.next_frame)

        # Exit cleanly
        self._app_quit_handler = t.add_handler('app quit', self._app_quit)
        self._close_cb = None
        
    def _get_position(self):
        # In independent desktop camera mode this is the desktop camera position,
        # otherwise it is the VR head mounted display position.
        return Camera.get_position(self)
    def _set_position(self, position):
        '''Move camera in scene while keeping camera in a fixed position in room.'''
        self.room_to_scene = position * self.position.inverse() * self.room_to_scene
        Camera.set_position(self, position)
        ui = self.user_interface
        if ui.shown():
            ui.move()
    position = property(_get_position, _set_position)

    @property
    def desktop_camera_position(self):
        '''Used for moving view with mouse when desktop camera is indpendent of vr camera.'''
        return self._desktop_camera_position if self.desktop_display == 'independent' else None
    
    def _get_room_to_scene(self):
        return self._room_to_scene
    def _set_room_to_scene(self, p):
        self._update_desktop_camera(p)
        self._room_to_scene = p
        self._reposition_user_interface()
    room_to_scene = property(_get_room_to_scene, _set_room_to_scene)

    def _update_desktop_camera(self, new_rts):
        if self.desktop_display != 'independent':
            return
        # Main camera stays at same position in room.
        rts = self._room_to_scene
        if rts is None:
            return   # VR room to scene not yet set.  Leave desktop camera unchanged.
        tf = new_rts * rts.inverse()
        mpos = tf * self._desktop_camera_position
        # Need to remove scale factor.
        x,y,z = tf.matrix[:,0]
        from math import sqrt
        s = 1/sqrt(x*x + y*y + z*z)
        m = mpos.matrix
        m[:3,:3] *= s
        from chimerax.core.geometry import Place
        self._desktop_camera_position = Place(m)
    
    def _move_camera_in_room(self, position):
        '''Move camera to given scene position without changing scene position in room.'''
        Camera.set_position(self, position)
        if self.initialize_desktop_camera_position:
            self._desktop_camera_position = position
            self.initialize_desktop_camera_position = False
        
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
        b = scene_bounds
        if b is None:
            b = self.vr_view.drawing_bounds()
        if b:
            scene_size = b.width()
            scene_center = b.center()
        else:
            scene_size = 1
            from numpy import zeros, float32
            scene_center = zeros((3,), float32)
        # First apply scene shift then scene scale to get room coords
        from chimerax.core.geometry import translation, scale
        from numpy import array, float32
        self.room_to_scene = (translation(scene_center) *
                              scale(scene_size/room_scene_size) *
                              translation(-array(room_center, float32)))
        
    def move_scene(self, move):
        '''Move is in room coordinates.'''
        self.room_to_scene = self.room_to_scene * move.inverse()
        for hc in self._hand_controllers:
            hc.update_scene_position(self)
        
    def _reposition_user_interface(self):
        ui = self.user_interface
        if ui.shown():
            ui.move()

    def close(self, close_cb = None):
        self._close = True
        self._close_cb = close_cb
        self._session.main_view.redraw_needed = True
        m = self._vr_model_group
        if m is not None:
            self._session.models.close([m])
            self._vr_model_group = None

    @property
    def closed(self):
        return self._close
    
    def _app_quit(self, tname, tdata):
        # On Linux (Ubuntu 18.04) the ChimeraX process does not exit
        # if VR has not been shutdown.
        import openvr
        openvr.shutdown()
        self._close = True	# Make sure openvr is not used any more.
        
    def _delayed_close(self):
        # Apparently OpenVR doesn't make its OpenGL context current
        # before deleting resources.  If the Qt GUI opengl context is current
        # openvr deletes the Qt resources instead.  So delay openvr close
        # until after rendering so that openvr opengl context is current.
        t = self._session.triggers
        t.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
        t.remove_handler(self._app_quit_handler)

        self._app_quit_handler = None
        for hc in self._hand_controllers:
            hc.close()
        self._hand_controllers = []
        
        self.user_interface.close()
        m = self._vr_model_group
        if m:
            if not m.deleted:
                self._session.models.close([m])
            self._vr_model_group = None
            
        td = self._texture_drawing
        if td is not None:
            td.delete()
            self._texture_drawing = None

        import openvr
        openvr.shutdown()
        self.vr_system = None
        self.compositor = None
        self._delete_framebuffers()
        if self._close_cb:
            self._close_cb()	# Replaces the main view camera and resets redraw rate.

    def _delete_framebuffers(self):
        fbs = self._framebuffers
        if fbs:
            self.render.make_current()
            for fb in fbs:
                fb.delete()
            self._framebuffers.clear()

    name = 'vr'
    '''Name of camera.'''
    
    @property
    def vr_view(self):
        return self._session.main_view

    @property
    def render(self):
        return self._session.main_view.render
    
    def _start_frame(self):
        if self._close:
            return
        c = self.compositor
        if c is None:
            return
        c.waitGetPoses(renderPoseArray = self._poses, gamePoseArray = None)
        self._frame_started = True

    def next_frame(self, *_):
        c = self.compositor
        if c is None or self._close:
            return

        self._start_frame()

        self.process_controller_events()
        self.user_interface.update_if_needed()

        # Get current headset position in room.
        import openvr
        hmd_pose0 = self._poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not hmd_pose0.bPoseIsValid:
            return
        H = hmd34_to_position(hmd_pose0.mDeviceToAbsoluteTracking) # head to room coordinates.
        
        # Compute camera scene position from HMD position in room
        from chimerax.core.geometry import scale
        S = scale(self.scene_scale)
        self.room_position = rp = H * S	# ChimeraX camera coordinates to room coordinates
        Cnew = self.room_to_scene * rp
        self._move_camera_in_room(Cnew)

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
                
    def process_controller_motion(self):

        for hc in self.hand_controllers():
            hc.process_motion(self)
        
    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        elif view_num == 2:
            v = self._desktop_camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            es = self.eye_shift_left if view_num == 0 else self.eye_shift_right
            t = es.scale_translation(1/self.scene_scale)
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        draw_desktop = (self.desktop_display == 'independent'
                        and self._session.ui.main_window.graphics_window.is_drawable)
        return 3 if draw_desktop else 2

    def view_width(self, point):
        fov = 100	# Effective field of view, degrees
        from chimerax.core.graphics.camera import perspective_view_width
        return perspective_view_width(point, self.position.origin(), fov)

    def view_all(self, bounds, window_size = None, pad = 0):
        fov = 100	# Effective field of view, degrees
        from chimerax.core.graphics.camera import perspective_view_all
        p = perspective_view_all(bounds, self.position, fov, window_size, pad)
        self._move_camera_in_room(p)
        self.fit_scene_to_room(bounds)

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        if view_num == 2:
            pixel_shift = (0,0)
            fov = self.desktop_field_of_view
            from chimerax.core.graphics.camera import perspective_projection_matrix
            return perspective_projection_matrix(fov, window_size, near_far_clip, pixel_shift)
        elif view_num == 0:
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
        left_fb, right_fb = self._eye_framebuffers(render)
        if view_num == 0:  # VR left-eye
            render.push_framebuffer(left_fb)
            render.mix_video = False
        elif view_num == 1:  # VR right-eye
            # Submit left eye texture (view 0) before rendering right eye (view 1)
            self._submit_eye_image('left', left_fb.openvr_texture, render)
            render.pop_framebuffer()
            render.push_framebuffer(right_fb)
        elif view_num == 2: # desktop view
            # Submit right eye texture (view 1) before rendering desktop (view 2)
            self._submit_eye_image('right', right_fb.openvr_texture, render)
            render.mix_video = True  # For making mixed reality videos
            render.mix_depth_scale = self.scene_scale

    def _submit_eye_image(self, side, texture, render):
        '''Side is "left" or "right".'''
        if self._close:
            return
        import openvr
        eye = openvr.Eye_Left if side == 'left' else openvr.Eye_Right
        # Caution: compositor.submit() changes the OpenGL read framebuffer binding to 0.
        result = self.compositor.submit(eye, texture)
        if self._use_opengl_flush:
            render.flush()
        self._check_for_compositor_error(side, result, render)

    def _check_for_compositor_error(self, eye, result, render):
        if result is not None:
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
        if self.number_of_views() == 2 and not self._close:
            rtex = render.current_framebuffer().openvr_texture
            self._submit_eye_image('right', rtex, render)

        render.pop_framebuffer()
        
        if self.desktop_display in ('mirror', 'independent'):
            # Render right eye to ChimeraX window.
            from chimerax.core.graphics.drawing import draw_overlays
            draw_overlays([self._desktop_drawing(render.render_size())], render)

        if self._close:
            self._delayed_close()

        self._frame_started = False

    def _eye_framebuffers(self, render):

        tw,th = self._render_size
        fbs = self._framebuffers
        if not fbs or fbs[0].width != tw or fbs[0].height != th:
            self._delete_framebuffers()
            from chimerax.core.graphics import Texture, opengl
            for eye in ('left', 'right'):
                t = Texture()
                t.initialize_rgba((tw,th))
                fb = opengl.Framebuffer('VR %s eye' % eye, render.opengl_context, color_texture = t)
                fbs.append(fb)
                # OpenVR texture id object
                import openvr
                fb.openvr_texture = ovrt = openvr.Texture_t()
                from ctypes import c_void_p
                ovrt.handle = c_void_p(int(t.id))
                ovrt.eType = openvr.TextureType_OpenGL
                ovrt.eColorSpace = openvr.ColorSpace_Gamma
        return fbs

    def _desktop_drawing(self, window_size):
        '''Used  to render ChimeraX desktop graphics window.'''
        td = self._texture_drawing
        if td is None:
            # Drawing object for rendering to ChimeraX window
            from chimerax.core.graphics.drawing import _texture_drawing
            t = self._framebuffers[1].color_texture
            self._texture_drawing = td = _texture_drawing(t)
            td.opaque_texture = True
        from chimerax.core.graphics.drawing import match_aspect_ratio
        match_aspect_ratio(td, window_size)
        return td

    def do_swap_buffers(self):
        return self.desktop_display != 'blank'

    def show_hand_controllers(self, show):
        self._controllers_show = show
        for hc in self._hand_controllers:
            hc.show_in_scene(show)

    def hand_controllers(self):
        self._check_if_controller_models_closed()
        cm = self._hand_controllers
        if len(cm) < 2:
            self._find_new_hand_controllers()
        return cm

    def _check_if_controller_models_closed(self):
        cm = self._hand_controllers
        cma =[hc for hc in cm if not hc._hand_model.deleted]
        if len(cma) < len(cm):
            self._hand_controllers = cma

    def _find_new_hand_controllers(self):
        # Check if a controller has been turned on.
        # Only check one controller id per-call to minimize performance penalty.
        import openvr
        d = self._controller_next_id
        self._controller_next_id = (d+1) % openvr.k_unMaxTrackedDeviceCount
        vrs = self.vr_system
        cm = self._hand_controllers
        if (vrs.getTrackedDeviceClass(d) == openvr.TrackedDeviceClass_Controller
            and vrs.isTrackedDeviceConnected(d)
            and d not in tuple(hc.device_index for hc in cm)):
            hc = HandController(d, self._session, vrs,
                                parent = self._vr_control_model_group(),
                                show = self._controllers_show)
            cm.append(hc)

    def _vr_control_model_group(self):
        g = self._vr_model_group
        if g is None or g.deleted:
            session = self._session
            g = Model('VR', session)
            g.SESSION_SAVE = False
            g.model_panel_show_expanded = False
            session.models.add([g], minimum_id = self._vr_model_group_id)
            self._vr_model_group = g
        return g
        
    def other_controller(self, controller):
        for hc in self.hand_controllers():
            if hc != controller:
                return hc
        return None

class UserInterface:
    '''
    Panel in VR showing ChimeraX main window.
    Buttons can be clicked with hand controllers.
    '''
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False

    def __init__(self, camera, session):
        self._camera = camera
        self._session = session

        self._mouse_mode_click_range = 5 # In scene units (Angstroms).
        self._update_later = 0		# Redraw panel after this many frames
        self._update_delay = 10		# After click on panel, update after this number of frames
        self._ui_model = None
        self._panels = []		# List of Panel, one for each user interface pane
        self._gui_tool_names = ['Toolbar', 'right panels']
        self._panel_y_spacing = 0.01	# meters
        self._panel_z_spacing = 0.001	# meters
        self._buttons_down = {}		# (HandController, button) -> Panel
        self._raised_buttons = {}	# maps highlight_id to (widget, panel)
        self._move_gui = set()		# set of (HandController, button) if gui being moved by press on title bar
        self._move_ui_mode = MoveUIMode()

        # Buttons that can be pressed on user interface.
        import openvr
        self.buttons = (openvr.k_EButton_SteamVR_Trigger, openvr.k_EButton_Grip, openvr.k_EButton_SteamVR_Touchpad,
                        openvr.k_EButton_A)
        
    def close(self):
        ui = self._ui_model
        if ui:
            if not ui.deleted:
                self._session.models.close([ui])
            self._ui_model = None

    @property
    def model(self):
        return self._ui_model

    @property
    def panels(self):
        return self._panels
    
    def shown(self):
        ui = self._ui_model
        if ui is None:
            return False
        if ui.deleted:
            self._ui_model = None
            return False
        return ui.display
    
    def show(self, room_position, parent_model):
        ui = self._ui_model
        if ui is None:
            self._ui_model = ui = self._create_ui_model(parent_model)
            self._panels = self._create_panels()
        self._update_ui_images()
        ui.room_position = room_position
        ui.position = self._camera.room_to_scene * room_position
        ui.display = True

    def _create_panels(self):
        ui = self._ui_model
        panels = []
        for tool_name in self._gui_tool_names:
            if (tool_name in ('right panels', 'main window')
                or _find_tool_by_name(tool_name, self._session)):
                p = Panel(ui, self, tool_name)
                panels.append(p)
            else:
                self._session.logger.warning('VR user interface could not find tool "%s"' % tool_name)
                
        np = len(panels)
        if np > 1:
            sep = self._panel_y_spacing
            dz = self._panel_z_spacing
            h = sum(p.size[1] for p in panels) + (np-1)*sep
            # Stack panels.
            y = h/2
            z = 0
            from chimerax.core.geometry import translation
            for p in panels:
                h = p.size[1]
                y -= 0.5*h
                pd = p._panel_drawing
                pd.position = translation((0,y,z)) * pd.position
                y -= 0.5*h + sep
                z -= dz
        return panels
    
    def move(self, room_motion = None):
        ui = self._ui_model
        if ui and ui.display:
            if room_motion:
                ui.room_position = room_motion * ui.room_position
            ui.position = self._camera.room_to_scene * ui.room_position            
        
    def hide(self):
        ui = self._ui_model
        if ui is not None:
            ui.display = False

    def set_gui_panels(self, tool_names):
        self._gui_tool_names = tool_names

    def process_hand_controller_button_event(self, hand_controller, button, pressed, released):
        b = button
        if b not in self.buttons:
            return False

        hc = hand_controller
        rp = hc.room_position
        if rp is None:
            return False

        bdown = self._buttons_down
        if released:
            if (hc,b) in bdown:
                # Current button down has been released.
                panel = bdown[(hc,b)]
                window_xy, z_offset = panel._panel_click_position(rp.origin())
                self._release(window_xy)
                del bdown[(hc,b)]
                return True
            elif (hc,b) in self._move_gui:
                self._move_gui.remove((hc,b))
                hc._drag_end(self._move_ui_mode, self._camera)
                return True
            else:
                # Button was released where we never got button press event.
                # For example button press away from panel, then release on panel.
                # Ignore release.
                return False
        elif pressed:
            # Button pressed.
            window_xy, panel = self._click_position(rp.origin())
            if panel:
                hand_mode = self._clicked_mouse_mode(window_xy)
                if hand_mode is not None:
                    self._enable_mouse_mode(hand_mode, hc, b, window_xy)
                elif self._clicked_on_title_bar(window_xy):
                    # Drag on title bar moves VR gui
                    self._move_gui.add((hc,b))
                    hc._drag_start(self._move_ui_mode, b, self._camera)
                    return True
                else:
                    self._press(window_xy)
                    bdown[(hc,b)] = panel
                    return True

        return False

    def _enable_mouse_mode(self, hand_mode, hand_controller, button, window_xy):
        if isinstance(hand_mode, MouseMode) and not hand_mode.has_vr_support:
            msg = 'No VR support for mouse mode %s' % hand_mode.name
        else:
            hand_controller._set_hand_mode(button, hand_mode)
            msg = 'VR mode %s' % hand_mode.name
        self._session.logger.info(msg)
        self._show_pressed(window_xy)
        self.redraw_ui()	# Show log message

    def process_hand_controller_motion(self, hand_controller):
        hc = hand_controller
        dragged = False
        for (bhc, b), panel in self._buttons_down.items():
            if hc == bhc:
                window_xy, z_offset = panel._panel_click_position(hc.room_position.origin())
                if window_xy is not None:
                    self._drag(window_xy)
                    dragged = True
        if dragged:
            return True

        # Highlight ui button under pointer
        self._highlight_button(hc.room_position.origin(), hc)

        return False

    def _press(self, window_xy):
        return self._click('press', window_xy)

    def _drag(self, window_xy):
        return self._click('move', window_xy)

    def _release(self, window_xy):
        return self._click('release', window_xy)

    def _click(self, type, window_xy):
        '''Type can be "press" or "release".'''
        w = self._post_mouse_event(type, window_xy)
        if w:
            if type == 'press':
                self._show_pressed_button(w)
            if type == 'release':
                self._show_pressed_button(w, pressed = False)
                self.redraw_ui()
            return True
        return False
    
    def _post_mouse_event(self, type, window_xy):
        '''Type is "press", "release" or "move".'''
        w, pos = self._clicked_widget(window_xy)
        if w is None or pos is None:
            return w
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
        return w
        
    def _clicked_widget(self, window_xy):
        ui = self._session.ui
        mw = ui.main_window
        from PyQt5.QtCore import QPoint, QPointF
        x,y = window_xy
        mwp = QPoint(int(x), int(y))
        w = mw.childAt(mwp)	# Works even if widget is covered.
        gp = mw.mapToGlobal(mwp)
        # Using w = ui.widgetAt(gp) does not work if the widget is covered by another app.
        wpos = QPointF(w.mapFromGlobal(gp)) if w else None
        return w, wpos

    def _show_pressed(self, window_xy, pressed = True):
        w, wpos = self._clicked_widget(window_xy)
        if w:
            self._show_pressed_button(w, pressed)

    def _show_pressed_button(self, widget, pressed = True):
        for w, panel in self._raised_buttons.values():
            if w == widget:
                widget._show_pressed = pressed
                panel._update_geometry()	# Show partially depressed button
            
    def _highlight_button(self, room_point, highlight_id):
        window_xy, panel = self._click_position(room_point)
        if panel:
            widget, wpos = self._clicked_widget(window_xy)
            from PyQt5.QtWidgets import QAbstractButton, QTabBar
            if isinstance(widget, (QAbstractButton, QTabBar)):
                rb = self._raised_buttons
                if highlight_id in rb and widget is rb[highlight_id]:
                    return # Already raised
                rb[highlight_id] = widget, panel
                panel._update_geometry()
                return

        rb = self._raised_buttons
        if highlight_id in rb:
            w, panel = rb[highlight_id]
            w._show_pressed = False
            del rb[highlight_id]
            panel._update_geometry()

    def redraw_ui(self, delay = True):
        if delay:
            self._update_later = self._update_delay
        else:
            self._update_later = 0
            self._update_ui_images()

    def update_if_needed(self):
        if self.shown() and self._update_later:
            self._update_later -= 1
            if self._update_later == 0:
                self._update_ui_images()

    def _update_ui_images(self):
        for panel in self._panels:
            panel._update_image()

    def _clicked_mouse_mode(self, window_xy):
        w, pos = self._clicked_widget(window_xy)
        from PyQt5.QtWidgets import QToolButton
        if isinstance(w, QToolButton):
            if hasattr(w, 'vr_mode'):
                if isinstance(w.vr_mode, str):
                    mouse_mode = self._session.ui.mouse_modes.named_mode(w.vr_mode)
                else:
                    mouse_mode = w.vr_mode()
                return self._hand_mode(mouse_mode)
            a = w.defaultAction()
            if hasattr(a, 'vr_mode'):
                mouse_mode = a.vr_mode()
                return self._hand_mode(mouse_mode)
        return None

    def _clicked_on_title_bar(self, window_xy):
        w, pos = self._clicked_widget(window_xy)
        from PyQt5.QtWidgets import QDockWidget
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        return isinstance(w, (QDockWidget, TabbedToolbar))
        
    def set_mouse_mode_click_range(self, range):
        self._mouse_mode_click_range = range

    def _hand_mode(self, mouse_mode):
        name = mouse_mode.name
        if name == 'zoom':
            m = ZoomMode()
        elif name in ('rotate', 'translate'):
            m = MoveSceneMode()
        else:
            m = MouseMode(mouse_mode, self._mouse_mode_click_range)
        return m
    
    def _click_position(self, room_point):
        if not self.shown():
            return None, None

        window_xy = panel = min_z_offset = None
        for p in self._panels:
            win_xy, z_offset = p._panel_click_position(room_point)
            if z_offset is not None and (min_z_offset is None or z_offset < min_z_offset):
                window_xy, panel, min_z_offset = win_xy, p, z_offset

        return window_xy, panel
    
    def _create_ui_model(self, parent):
        ses = self._session
        from chimerax.core.models import Model
        m = Model('User interface', ses)
        m.skip_bounds = True
        m.color = (255,255,255,255)
        m.use_lighting = False
        ses.models.add([m], parent = parent)
        return m

    def display_ui(self, hand_room_position, camera_position):
        rp = hand_room_position
        # Orient horizontally and facing camera.
        view_axis = camera_position.origin() - rp.origin()
        from chimerax.core.geometry import orthonormal_frame, translation
        p = orthonormal_frame(view_axis, (0,1,0), origin = rp.origin())
        # Offset vertically
        # p = translation(0.5 * width * p.axes()[1]) * p
        parent = self._camera._vr_control_model_group()
        self.show(p, parent)

    def scale_ui(self, scale_factor):
        from numpy import mean
        center = mean([p._panel_drawing.position.origin() for p in self._panels], axis = 0)
        for p in self._panels:
            p.scale_panel(scale_factor, center)

class Panel:
    '''The VR user interface consists of one or more rectangular panels.'''
    initial_size = {'main window': (1,1), 'right panels': (0.5,1), 'Toolbar': (1,.1)} # Meters
    def __init__(self, parent, ui, tool_name = 'main window'):
        self._ui = ui
        self._gui_tool_name = tool_name	# Name of tool instance shown in VR gui panel.
        self._max_size = Panel.initial_size.get(tool_name, (0.5,1))  # Room coords, meters
        x0,y0,w,h = self._panel_rectangle()
        self._size = self._fit_size(w,h) # Billboard width, height in room coords, meters.
        self._panel_size = None 	# Panel size in Qt device independent pixels
        self._panel_offset = (0,0)  	# Offset from desktop main window upper left corner, to panel rectangle in Qt device independent pixels
        self._last_image_rgba = None
        self._ui_click_range = 0.05 	# Maximum distance of click from plane, room coords, meters.
        self._button_rise = 0.01	# meters rise when pointer over button

        # Drawing that renders this panel.
        self._panel_drawing = self._create_panel_drawing(parent)

    def _create_panel_drawing(self, parent):
        from chimerax.core.graphics import Drawing
        d = Drawing('VR UI panel')
        d.color = (255,255,255,255)
        d.use_lighting = False
        d.skip_bounds = True
        parent.add_drawing(d)
        return d

    @property
    def name(self):
        return self._gui_tool_name

    @property
    def size(self):
        '''Panel width and height in meters.'''
        return self._size

    def _fit_size(self, w, h):
        # Inscribe rectangle with aspect (w,h) in self._max_size rectangle.
        mw, mh = self._max_size
        if w == 0 and h == 0:
            return (0,0)
        size = ((w/h)*mh, mh) if mw*h >= mh*w else (mw, (h/w)*mw)
        return size

    @property
    def drawing(self):
        return self._panel_drawing
    
    def scale_panel(self, scale_factor, center = None):
        '''
        Center is specified in the parent model coordinate system.
        If center is not specified then panel scales about its geometric center.
        '''
        w,h = self.size
        self._size = (scale_factor*w, scale_factor*h)
        mw,mh = self._max_size
        self._max_size = (scale_factor*mw, scale_factor*mh)

        self._update_geometry()

        if center is not None:
            pd = self._panel_drawing
            shift = (scale_factor-1) * (pd.position.origin() - center)
            from chimerax.core.geometry import translation
            pd.position = translation(shift) * pd.position
            
    def _panel_click_position(self, room_point):
        ui = self._panel_drawing
        scene_point = self._ui._camera.room_to_scene * room_point
        x,y,z = ui.scene_position.inverse() * scene_point
        w,h = self.size
        hw, hh = 0.5*w, 0.5*h
        cr = self._ui_click_range
        on_panel = (x >= -hw and x <= hw and y >= -hh and y <= hh and z >= -cr and z <= cr)
        z_offset = (z - cr) if on_panel else None
        sx, sy = self._panel_size
        ox, oy = self._panel_offset
        ws = 1/w if w > 0 else 0
        hs = 1/h if h > 0 else 0
        window_xy = ox + sx * (x + hw) * ws, oy + sy * (hh - y) * hs
        return window_xy, z_offset

    def _update_image(self):
        rgba = self._panel_image()
        lrgba = self._last_image_rgba
        self._last_image_rgba = rgba
        if lrgba is None or rgba.shape != lrgba.shape:
            h,w = rgba.shape[:2]
            self._size = self._fit_size(w,h)
            self._update_geometry()

        d = self._panel_drawing
        if d.texture is not None:
            # Require OpenGL context for deleting texture.
            self._ui._session.main_view.render.make_current()
            d.texture.delete_texture()
        from chimerax.core.graphics import Texture
        d.texture = Texture(rgba)

    def _update_geometry(self):
        # Vertex coordinates are in room coordinates (meters), and
        # position matrix contains scale factor to produce scene coordinates.

        # Calculate rectangles for panel and raised buttons
        w, h = self.size
        xmin,ymin,xmax,ymax = -0.5*w,-0.5*h,0.5*w,0.5*h
        rects = [(xmin,ymin,0,xmax,ymax,0)]
        zr = self._button_rise
        rb = self._ui._raised_buttons
        for widget, panel in rb.values():
            if panel is self:
                x0,y0,x1,y1 = self._button_rectangle(widget)
                z = .5*zr if getattr(widget, '_show_pressed', False) else zr
                rects.append((x0,y0,z,x1,y1,z))

        # Create geometry for rectangles
        nr = len(rects)
        nv = 4*nr
        nt = 2*nr
        from numpy import empty, float32, int32
        v = empty((nv,3), float32)
        tc = empty((nv,2), float32)
        t = empty((nt,3), int32)
        for r, (x0,y0,z0,x1,y1,z1) in enumerate(rects):
            ov, ot = 4*r, 2*r
            v[ov:ov+4] = ((x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0))
            ws = 1/w if w > 0 else 0
            hs = 1/h if h > 0 else 0
            tx0, ty0, tx1, ty1 = (x0-xmin)*ws, (y0-ymin)*hs, (x1-xmin)*ws, (y1-ymin)*hs
            tc[ov:ov+4] = ((tx0,ty0), (tx1,ty0), (tx1,ty1), (tx0,ty1))
            t[ot:ot+2] = ((ov,ov+1,ov+2), (ov,ov+2,ov+3))

        # Update Drawing
        d = self._panel_drawing
        d.set_geometry(v, None, t)
        d.texture_coordinates = tc

    def panel_image_rgba(self):
        return self._last_image_rgba

    def set_gui_tool_name(self, tool_name):
        self._gui_tool_name = tool_name

    def _panel_image(self):
        ui = self._ui._session.ui
        im = ui.window_image()
        from chimerax.core.graphics.drawing import qimage_to_numpy
        rgba = qimage_to_numpy(im)
        wh,ww = rgba.shape[:2]
        dpr = ui.main_window.devicePixelRatio()
        x0, y0, w, h = self._panel_rectangle()
        prgba = rgba[wh-dpr*(y0+h):wh-dpr*y0,dpr*x0:dpr*(x0+w),:]
        self._panel_offset = (x0, y0)
        self._panel_size = (w, h)
        return prgba

    def _panel_rectangle(self):
        '''
        Returned coordinates are in pixels relative to the top level window.
        A y value of zero is at the top with increasing y values going down on screen.
        '''
        tname = self._gui_tool_name
        mw = self._ui._session.ui.main_window
        if tname == 'main window':
            # Show entire main window in VR.
            x0, y0, w, h = 0, 0, mw.width(), mw.height()
        elif tname == 'right panels':
            gw = mw.graphics_window
            x0, y0 = gw.x() + gw.width(), gw.y()
            h = gw.height()
            w = mw.width() - x0
        else:
            tw = self._gui_tool_window()
            if tw:
                x0, y0, w, h  = tw.x(), tw.y(), tw.width(), tw.height()
            else:
                self._ui._session.logger.warning('Tool panel "%s" for VR gui was not found' % tname)
                x0, y0, w, h = 0, 0, mw.width(), mw.height()
# TODO: The x,y coords need to be in pixels relative to the ChimeraX main window.
#       But the code gw.x() or tw.x() gives the corner position relative to the parent
#       window which may not be the top level window.
        return x0, y0, w, h

    def _button_rectangle(self, widget):
        '''Returns coordinates in meters with 0,0 at center of ui panel.'''
        mw = self._ui._session.ui.main_window
        x0,y0,w,h = self._panel_rectangle()
        xc, yc = x0 + 0.5*w, y0 + 0.5*h
        from PyQt5.QtCore import QPoint
        wxy0 = widget.mapTo(mw, QPoint(0,0))
        wx0,wy0 = wxy0.x(), wxy0.y()
        ww,wh = widget.width(), widget.height()
        wx1, wy1 = wx0+ww, wy0+wh
        pw, ph = self.size
        ws = 1/w if w > 0 else 0
        hs = 1/h if h > 0 else 0
        rect = (pw*(wx0-xc)*ws, -ph*(wy0-yc)*hs, pw*(wx1-xc)*ws, -ph*(wy1-yc)*hs)
        return rect

    def _gui_tool_window(self):
        tname = self._gui_tool_name
        if tname is None:
            return None

        ti = _find_tool_by_name(tname, self._ui._session)
        return ti.tool_window._dock_widget if ti else None

def _find_tool_by_name(name, session):
    for ti in session.tools.list():
        if ti.tool_name == name and hasattr(ti, 'tool_window') and ti and ti.displayed:
            return ti
    return None
        
class HandController:
    _controller_colors = ((200,200,0,255), (0,200,200,255))

    def __init__(self, device_index, session, vr_system,
                 parent, show = True, length = 0.20, radius = 0.04):

        self.device_index = device_index
        self.vr_system = vr_system

        from openvr import Prop_RenderModelName_String
        model_name = vr_system.getStringTrackedDeviceProperty(device_index, Prop_RenderModelName_String)
        # 'vr_controller_vive_1_5' for vive pro
        # 'oculus_cv1_controller_right', 'oculus_cv1_controller_left'
        # 'oculus_rifts_controller_right', 'oculus_rifts_controller_left'
        self._controller_type = model_name
        
        # Create hand model
        name = 'Hand %s' % device_index
        self._hand_model = hm = HandModel(session, name, length=length, radius=radius,
                                          color = self._cone_color(), controller_type = self._controller_type)
        hm.display = show
        parent.add([hm])

        # Assign actions bound to controller buttons
        self._modes = {}			# Maps button name to HandMode
        self._active_drag_modes = set() # Modes with an active drag (ie. button down and not yet released).
        oculus = self._controller_type.startswith('oculus')
        grip_mode = MoveSceneMode() if oculus else RecenterMode()
        import openvr
        initial_modes = [(openvr.k_EButton_Grip, grip_mode),
                         (openvr.k_EButton_ApplicationMenu, ShowUIMode()),
                         (openvr.k_EButton_SteamVR_Trigger, MoveSceneMode()),
                         (openvr.k_EButton_SteamVR_Touchpad, ZoomMode()),
        ]
        if oculus:
            initial_modes.append((openvr.k_EButton_A, ZoomMode()))
        # Oculus touch controller left and right buttons:
        #    trigger = k_EButton_Axis1 = 33 = k_EButton_SteamVR_Trigger
        #    grip = k_EButton_Grip = 2 and k_EButton_Axis2 = 34 both
        #    A or X button = k_EButton_A = 7
        #    B or Y button = k_EButton_ApplicationMenu = 1
        #    thumbstick = k_EButton_Axis0 = 32 = k_EButton_SteamVR_Touchpad

        for button, mode in initial_modes:
            self._set_hand_mode(button, mode)
    
    def _cone_color(self):
        cc = self._controller_colors
        from numpy import array, uint8
        rgba8 = array(cc[self.device_index%len(cc)], uint8)
        return rgba8

    @property
    def room_position(self):
        return self._hand_model.room_position

    @property
    def tip_room_position(self):
        return self._hand_model.room_position.origin()

    @property
    def position(self):
        return self._hand_model.position

    @property
    def button_modes(self):
        return self._modes

    @property
    def left_or_right(self):
        import openvr
        left_id = self.vr_system.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_LeftHand)
        return 'left' if self.device_index == left_id else 'right'
    
    def close(self):
        hm = self._hand_model
        if hm:
            if not hm.deleted:
                hm.session.models.close([hm])
            self._hand_model = None

    def show_in_scene(self, show):
        self._hand_model.display = show
        
    def _update_position(self, camera):
        '''Move hand controller model to new position.
        Keep size constant in physical room units.'''
        dp = camera._poses[self.device_index].mDeviceToAbsoluteTracking
        self._hand_model.room_position = hmd34_to_position(dp)
        self.update_scene_position(camera)

    def update_scene_position(self, camera):
        hm = self._hand_model
        hm.position = camera.room_to_scene * hm.room_position
            
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
        self._hand_model._show_button_down(b, pressed)
        m = self._modes.get(b)
        if not isinstance(m, ShowUIMode):
            # Check for click on UI panel.
            ui = camera.user_interface
            if ui.process_hand_controller_button_event(self, b, pressed, released):
                return
        
        # Call HandMode press() or release() callback.
        if m:
            if pressed:
                self._drag_start(m, b, camera)
            else:
                self._drag_end(m, camera)

    def _drag_start(self, mode, button, camera):
        mode.pressed(camera, self)
        mode._button_down = button
        self._active_drag_modes.add(mode)

    def _drag_end(self, mode, camera):
        mode.released(camera, self)
        self._active_drag_modes.discard(mode)
        if not isinstance(mode, (ShowUIMode, MoveSceneMode, ZoomMode)):
            camera.user_interface.redraw_ui()
        
    def _set_hand_mode(self, button, hand_mode):
        self._modes[button] = hand_mode
        self._hand_model._set_button_icon(button, hand_mode.icon_path)

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
        
    def process_motion(self, camera):
        # Move hand controller model
        previous_pose = self.room_position
        self._update_position(camera)

        # Generate mouse move event on ui panel.
        ui = camera.user_interface
        if ui.process_hand_controller_motion(self):
            return	# UI drag in progress.

        # Do hand controller drag when buttons pressed
        if previous_pose is not None:
            self._check_for_missing_button_release(camera)
            pose = self.room_position
            for m in self._active_drag_modes:
                m.drag(camera, self, previous_pose, pose)

    def _check_for_missing_button_release(self, camera):
        '''Cancel drag modes if button has been released even if we didn't get a button up event.'''
        adm = self._active_drag_modes
        if len(adm) == 0:
            return
        success, cstate = self.vr_system.getControllerState(self.device_index)
        if success:
            pressed_mask = cstate.ulButtonPressed
            for m in tuple(adm):
                # bm = openvr.ButtonMaskFromId(m._button_down)  # Routine is missing from pyopenvr
                bm = 1 << m._button_down
                if not pressed_mask & bm:
                    self._drag_end(m, camera)
        
from chimerax.core.models import Model
class HandModel(Model):
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False

    def __init__(self, session, name, length = 0.20, radius = 0.04, color = (200,200,0,255),
                 controller_type = 'htc vive'):
        Model.__init__(self, name, session)

        from chimerax.core.geometry import Place
        self.room_position = Place()	# Hand controller position in room coordinates.

        self._cone_color = color
        self._button_color = (255,255,255,255)	# White
        self.color = (255,255,255,255)	# Texture modulation color

        self._controller_type = controller_type
        
        # Avoid hand disappearing when behind models, especially in multiperson VR.
        self.allow_depth_cue = False
        
        # Draw controller as a cone.
        self._create_model_geometry(length, radius, color)

    def _create_model_geometry(self, length, radius, color, tex_size = 160):
        from chimerax.surface.shapes import cone_geometry
        cva, cna, cta = cone_geometry(nc = 50, points_up = False)
        self._num_cone_vertices = len(cva)
        from numpy import empty, float32
        ctc = empty((len(cva), 2), float32)
        ctc[:] = .5/tex_size
        cva[:,:2] *= radius
        cva[:,2] += 0.5		# Move tip to 0,0,0 for picking
        cva[:,2] *= length
        geom = [(cva,cna,ctc,cta)]

        self._buttons = b = HandButtons(self._controller_type)
        geom.extend(b.geometry(length, radius))
        from chimerax.core.graphics import concatenate_geometry
        va, na, tc, ta = concatenate_geometry(geom)
        
        self._cone_vertices = va
        self.set_geometry(va, na, ta)
        self.texture_coordinates = tc

        # Button icons texture
        self.texture = b.texture(self._cone_color, self._button_color, tex_size)

    def set_cone_color(self, color):
        if color != self._cone_color:
            self._cone_color = color
            self._buttons.set_cone_color(color)
        
    def _show_button_down(self, b, pressed):
        cv = self._cone_vertices
        vbuttons = cv[self._num_cone_vertices:]
        self._buttons.button_vertices(b, pressed, vbuttons)
        self.set_geometry(cv, self.normals, self.triangles)
        
    def _set_button_icon(self, button, icon_path):
        self._buttons.set_button_icon(button, icon_path)

def hand_mode_icon_path(session, mode_name):
    if mode_name == 'recenter':
        return RecenterMode.icon_location()
    elif mode_name == 'move scene':
        return MoveSceneMode.icon_location()
    elif mode_name == 'show ui':
        return ShowUIMode.icon_location()
    else:
        for mm in session.ui.mouse_modes.modes:
            if mm.name == mode_name:
                return mm.icon_path
    return None

class HandButtons:
    def __init__(self, controller_type = 'htc vive'):
        # Cone buttons
        import openvr
        buttons = [
        ]
        if controller_type.startswith('oculus'):
            side, thumb_side, menu_side = (180,110,140) if controller_type.endswith('right') else (0,70,40)
            buttons = [
                ButtonGeometry(openvr.k_EButton_SteamVR_Trigger, z=.4, radius=.01, azimuth=270, tex_range=(.2,.4)),
                ButtonGeometry(openvr.k_EButton_A, z=.5, radius=.01, azimuth=thumb_side, tex_range=(.4,.6)),
                ButtonGeometry(openvr.k_EButton_Grip, z=.6, radius=.01, azimuth=side, tex_range=(.6,.8)),
                ButtonGeometry(openvr.k_EButton_ApplicationMenu, z=.4, radius=.006, azimuth=menu_side, tex_range=(.8,1)),
            ]
        else:
            # Vive controllers
            buttons = [
                ButtonGeometry(openvr.k_EButton_SteamVR_Trigger, z=.5, radius=.01, azimuth=270, tex_range=(.2,.4)),
                ButtonGeometry(openvr.k_EButton_SteamVR_Touchpad, z=.5, radius=.01, azimuth=90, tex_range=(.4,.6)),
                ButtonGeometry(openvr.k_EButton_Grip, z=.7, radius=.01, azimuth=0, tex_range=(.6,.8)),
                ButtonGeometry(openvr.k_EButton_Grip, z=.7, radius=.01, azimuth=180, tex_range=(.6,.8)),
                ButtonGeometry(openvr.k_EButton_ApplicationMenu, z=.35, radius=.006, azimuth=90, tex_range=(.8,1)),
            ]
                
        self._buttons = buttons
        self._texture = None
        self._icon_scale = .8	# Scaled image centered in square circumscribing circular button

    def geometry(self, length, radius):
        return [b.cone_button_geometry(length, radius) for b in self._buttons]

    def texture(self, cone_color, button_color, tex_size):
        nb = len(set([b.button for b in self._buttons]))
        from numpy import empty, uint8
        self._button_rgba = rgba = empty((tex_size, tex_size*(nb + 1),4), uint8)
        rgba[:,0:tex_size,:] = cone_color
        rgba[:,tex_size:,:] = button_color
        from chimerax.core.graphics import Texture
        self._texture = t = Texture(rgba)
        return t

    def set_cone_color(self, color):
        t = self._texture
        rgba = self._button_rgba
        if t is not None and rgba is not None:
            tex_size = rgba.shape[0]
            rgba[:,0:tex_size,:] = color
            t.reload_texture(rgba)
            
    def _button_geometry(self, button):
        for b in self._buttons:
            if b.button == button:
                return b
        return None
    
    def set_button_icon(self, button, icon_path):
        bg = self._button_geometry(button)
        if bg:
            rgba = self._button_rgba
            icon_size = int(self._icon_scale * rgba.shape[0])
            bg.set_icon_image(self._button_rgba, icon_path, icon_size)
            self._texture.reload_texture(rgba)

    def button_vertices(self, button, lowered, vertices):
        voffset = 0
        for b in self._buttons:
            if b.button == button:
                v = b.vertices_lowered if lowered else b.vertices_raised
                vertices[voffset:voffset+len(v)] = v
            voffset += b.num_vertices
    
class ButtonGeometry:
    def __init__(self, button, z, radius, azimuth, tex_range, rise = 0.002, num_vertices = 30):
        '''
        z is button center position from cone tip at 0 to base at 1.
        radius is in meters
        azimuth is in degrees, 90 on top, 270 bottom.
        tex_range is u texture coordinate range for mapping icon onto button.
        rise is height above cone surface in meters.
        '''
        self.button = button
        self.z = z
        self.radius = radius
        self.azimuth = azimuth
        self.tex_range = tex_range
        self.rise = rise
        self.num_vertices = num_vertices

    def cone_button_geometry(self, cone_length, cone_radius):
        '''
        Map circular disc onto cone surface.  Cone axis is z axis pointing down with tip at origin,
        cone height 1, cone base radius, disc center fraction f from cone tip to base edge.
        Disc is raised along normals above cone surface by rise.  Disc perimeter is defined
        by n vertices (must be even).  Texture coordinates range is (umin, umax, vmin, vmax).
        Return vertices, normals and triangles.
        '''
        cl,cr = cone_length, cone_radius
        from math import sqrt, sin, cos, pi, atan2
        e = sqrt(cr*cr+cl*cl)
        sca = cr/e  # sin(cone_angle)
        cca = cl/e  # cos(cone_angle)
        y0 = self.z * e
        aoffset = self.azimuth * pi/180
        from numpy import empty, float32, int32
        n = self.num_vertices
        va = empty((n,3), float32)
        na = empty((n,3), float32)
        tc = empty((n,2), float32)
        u0,u1 = self.tex_range[::-1]
        v0,v1 = 1,0
        for i in range(n):
            a = 2*pi*i/n
            ca, sa = cos(a), sin(a)
            x,y = self.radius*ca, y0 + self.radius*sa
            az = aoffset + atan2(x,y)/sca
            r = sqrt(x*x + y*y)
            va[i,:] = (r*sca*cos(az), r*sca*sin(az), r*cca)
            na[i,:] = (cca*cos(az), cca*sin(az), -sca)
            tc[i,:] = (u0+(u1-u0)*0.5*(1+ca), v0+(v1-v0)*0.5*(1+sa))

        self.vertices_lowered = va + 0.1*self.rise*na
        va += self.rise*na
        self.vertices_raised = va.copy()

        ta = empty((n-2,3), int32)
        for i in range(n//2-1):
            ta[2*i,:] = (i, i+1, n-1-i)
            ta[2*i+1,:] = (i+1, n-2-i, n-1-i)

        return va, na, tc, ta

    def set_icon_image(self, tex_rgba, icon_path, image_size):
        if icon_path is None:
            return
        from PyQt5.QtGui import QImage
        qi = QImage(icon_path)
        s = image_size
        if qi.width() != s or qi.height() != s:
            qi = qi.scaled(s,s)
        from chimerax.core.graphics import qimage_to_numpy
        rgba = qimage_to_numpy(qi)
        # TODO: Need to alpha blend with button background.
        transp = (rgba[:,:,3] == 0)
        from numpy import putmask
        for c in range(4):
            putmask(rgba[:,:,c], transp, 255)
        tsize = tex_rgba.shape[0]
        inset = (tsize - rgba.shape[0]) // 2
        i0 = inset
        i1 = i0 + rgba.shape[0]
        j0 = int(self.tex_range[0] * tex_rgba.shape[1]) + inset
        j1 = j0+rgba.shape[1]
        tex_rgba[i0:i1,j0:j1,:] = rgba
    
class HandMode:
    @property
    def icon_path(self):
        return None
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

class MoveUIMode(HandMode):
    name = 'move ui'
    def __init__(self):
        self._last_hand_position = {}	# HandController -> Place
        HandMode.__init__(self)
    def pressed(self, camera, hand_controller):
        self._last_hand_position[hand_controller] = hand_controller.room_position
    def released(self, camera, hand_controller):
        self._last_hand_position[hand_controller] = None
    def drag(self, camera, hand_controller, previous_pose, pose):
        ui = camera.user_interface
        oc = camera.other_controller(hand_controller)
        if oc and self._ui_zoom(oc):
            scale, center = _pinch_scale(previous_pose.origin(), pose.origin(), oc.tip_room_position)
            ui.scale_ui(scale)
            self._last_hand_position.clear()	# Avoid jump when one button released
        else:
            hrp = hand_controller.room_position
            lhrp = self._last_hand_position.get(hand_controller)
            if lhrp is not None:
                ui.move(hrp * lhrp.inverse())
            self._last_hand_position[hand_controller] = hrp
    def _ui_zoom(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, MoveUIMode):
                return True
        return False

class ShowUIMode(MoveUIMode):
    name = 'show ui'
    def __init__(self):
        self._start_ui_move_time = None
        self._ui_hide_time = 0.3	# seconds. Max application button press/release time to hide ui
        MoveUIMode.__init__(self)
    @property
    def icon_path(self):
        return ShowUIMode.icon_location()
    @staticmethod
    def icon_location():
        from os.path import join, dirname
        return join(dirname(__file__), 'menu_icon.png')
    def pressed(self, camera, hand_controller):
        ui = camera.user_interface
        if ui.shown():
            from time import time
            self._start_ui_move_time = time()
        else:
            ui.display_ui(hand_controller.room_position, camera.room_position)
        MoveUIMode.pressed(self, camera, hand_controller)
    def released(self, camera, hand_controller):
        # End UI move, or hide.
        stime = self._start_ui_move_time
        from time import time
        if stime is not None and time() < stime + self._ui_hide_time:
            camera.user_interface.hide()
        self._start_ui_move_time = None
        MoveUIMode.released(self, camera, hand_controller)

class MoveSceneMode(HandMode):
    name = 'move scene'
    def __init__(self):
        self._zoom_center = None
    @property
    def icon_path(self):
        return MoveSceneMode.icon_location()

    @staticmethod
    def icon_location():
        from chimerax.mouse_modes import TranslateMouseMode
        return TranslateMouseMode.icon_location()

    def drag(self, camera, hand_controller, previous_pose, pose):
        oc = camera.other_controller(hand_controller)
        if oc and self._other_controller_move(oc):
            # Both controllers trying to move scene -- zoom
            scale, center = _pinch_scale(previous_pose.origin(), pose.origin(), oc.tip_room_position)
            if self._zoom_center is None:
                self._zoom_center = _choose_zoom_center(camera, center)
            _pinch_zoom(camera, scale, self._zoom_center)
        else:
            self._zoom_center = None
            move = pose * previous_pose.inverse()
            camera.move_scene(move)

    def released(self, camera, hand_controller):
        self._zoom_center = None
        
    def _other_controller_move(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, MoveSceneMode):
                return True
        return False

def _pinch_scale(prev_pos, pos, other_pos):
    from chimerax.core.geometry import distance
    d, dp = distance(pos,other_pos), distance(prev_pos,other_pos)
    if dp > 0:
        s = d / dp
        s = max(min(s, 10.0), 0.1)	# Limit scaling
    else:
        s = 1.0
    center = 0.5*(pos+other_pos)
        
    return s, center

class ZoomMode(HandMode):
    name = 'zoom'
    size_doubling_distance = 0.1	# meters, vertical motion
    def __init__(self):
        self._zoom_center = None
        self._use_scene_center = False
    @property
    def icon_path(self):
        return ZoomMode.icon_location()
    @staticmethod
    def icon_location():
        from chimerax.mouse_modes import ZoomMouseMode
        return ZoomMouseMode.icon_location()
    def pressed(self, camera, hand_controller):
        self._zoom_center = _choose_zoom_center(camera, hand_controller.tip_room_position)
    def drag(self, camera, hand_controller, previous_pose, pose):
        center = self._zoom_center
        if center is None:
            return
        y_motion = (pose.origin() - previous_pose.origin())[1]  # meters
        s = 2 ** (y_motion/self.size_doubling_distance)
        scale_factor = max(min(s, 10.0), 0.1)	# Limit scaling
        _pinch_zoom(camera, scale_factor, center)
    def released(self, camera, hand_controller):
        self._zoom_center = None

def _choose_zoom_center(camera, center):
    # Zoom in about center of scene if requested center point is outside scene bounding box.
    # This avoids pushing a distant scene away.
    b = camera.vr_view.drawing_bounds()
    if b and not b.contains_point(camera.room_to_scene * center):
        return camera.room_to_scene.inverse() * b.center()
    return center

def _pinch_zoom(camera, scale_factor, center):
    from chimerax.core.geometry import distance, translation, scale
    scale = translation(center) * scale(scale_factor) * translation(-center)
    camera.move_scene(scale)

class RecenterMode(HandMode):
    name = 'recenter'
    def pressed(self, camera, hand_controller):
        camera.fit_scene_to_room()
    @property
    def icon_path(self):
        return self.icon_location()
    @staticmethod
    def icon_location():
        from os.path import join, dirname
        from chimerax import shortcuts
        return join(dirname(shortcuts.__file__), 'icons', 'viewall.png')

class MouseMode(HandMode):
    name = 'mouse mode'
    def __init__(self, mouse_mode, click_range = 5.0):
        self._mouse_mode = mouse_mode
        mouse_mode.enable()
        self.name = mouse_mode.name
        self._last_drag_room_position = None # Hand controller position at last vr_motion call
        self._laser_range = click_range	# Range for mouse mode laser clicks in scene units (Angstroms)

    @property
    def has_vr_support(self):
        m = self._mouse_mode
        return hasattr(m, 'vr_press') or hasattr(m, 'vr_motion') or hasattr(m, 'vr_release')

    @property
    def icon_path(self):
        return self._mouse_mode.icon_path
    
    def pressed(self, camera, hand_controller):
        self._click(camera, hand_controller, True)

    def released(self, camera, hand_controller):
        self._click(camera, hand_controller, False)

    def _click(self, camera, hand_controller, pressed):
        m = self._mouse_mode
        if hasattr(m, 'vr_press') and pressed:
            p = hand_controller.position
            xyz1 = p * (0,0,0)
            range_scene = self._laser_range
            xyz2 = p * (0,0,-range_scene)
            m.vr_press(xyz1, xyz2)
        if hasattr(m, 'vr_motion'):
            self._last_drag_room_position = hand_controller.room_position if pressed else None
        if hasattr(m, 'vr_release') and not pressed:
            m.vr_release()

    def drag(self, camera, hand_controller, previous_pose, pose):
        m = self._mouse_mode
        if hasattr(m, 'vr_motion'):
            rp = hand_controller.room_position
            ldp = self._last_drag_room_position
            room_move = rp * ldp.inverse()
            delta_z = (rp.origin() - ldp.origin())[1] # Room vertical motion
            rts = camera.room_to_scene
            move = rts * room_move * rts.inverse()
            p = rts * rp
            if m.vr_motion(p, move, delta_z) != 'accumulate drag':
                self._last_drag_room_position = rp
            
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
    
