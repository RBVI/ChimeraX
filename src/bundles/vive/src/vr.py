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
def vr(session, enable = None, room_position = None, mirror = None,
       gui = None, center = None, click_range = None,
       multishadow_allowed = False, simplify_graphics = True):
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
      Controls whether VR scene is mirrored to the desktop display graphics window.
      Default true.
    gui : string
      Name of a tool instance which will be shown as the VR gui panel.  If not specified
      then the VR gui panel consists of all tools docked on the right side of the main window.
    center : bool
      Whether to center and scale models to fit in room.  This is always done the first time VR
      is started.  If vr is turned off and the on it remembers the previous model position unless
      this options is specified.
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
    '''
    
    if enable is None and room_position is None:
        enable = True

    c = vr_camera(session)
    start = (session.main_view.camera is not c)

    if enable is not None:
        if enable:
            start_vr(session, multishadow_allowed, simplify_graphics)
        else:
            stop_vr(session, simplify_graphics)

    if room_position is not None:
        if isinstance(room_position, str) and room_position == 'report':
            p = ','.join('%.5g' % x for x in tuple(c.room_to_scene.matrix.flat))
            session.logger.info(p)
        else:
            c.room_to_scene = room_position

    if mirror is None and start:
        if not wait_for_vsync(session, False):
            session.logger.warning('Graphics on desktop display may cause VR to flicker.'
                                   '  Turning off mirroring to desktop display.')
            mirror = False
    if mirror:
        if not wait_for_vsync(session, False):
            session.logger.warning('Graphics on desktop display may cause VR to flicker.')
        c.mirror = mirror

    if gui is not None:
        c.user_interface.set_gui_panels([tool_name.strip() for tool_name in gui.split(',')])

    if center:
        c.fit_scene_to_room()
        
    if click_range is not None:
        c.user_interface.set_mouse_mode_click_range(click_range)

# -----------------------------------------------------------------------------
# Assign VR hand controller buttons
#
def vr_button(session, button, mode, hand = None):
    '''
    Assign VR hand controller buttons

    Parameters
    ----------
    button : 'trigger', 'grip', 'touchpad', 'thumbstick', 'menu', 'A', 'B', 'X', 'Y', 'all'
      Name of button to assign.  Buttons A/B are for Oculus controllers and imply hand = 'right',
      and X/Y imply hand = 'left'
    mode : HandMode instance or 'default'
      VR hand mode to assign to button.
    hand : 'left', 'right', None
      Which hand controller to assign.  If None then assign button on both hand controllers.
      If button is A, B, X, or Y then hand is ignored since A/B implies right and X/Y implies left.
    '''

    c = vr_camera(session)

    if button in ('A', 'B'):
        hand = 'right'
    elif button in ('X', 'Y'):
        hand = 'left'
        
    hclist = [hc for hc in c.hand_controllers() if hand is None or hc.left_or_right == hand]
    if len(hclist) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Hand controller is not enabled.')

    from openvr import \
        k_EButton_Grip as grip, \
        k_EButton_ApplicationMenu as menu, \
        k_EButton_SteamVR_Trigger as trigger, \
        k_EButton_SteamVR_Touchpad as touchpad, \
        k_EButton_A as a
    
    openvr_buttons = {
        'grip': [grip],
        'menu': [menu],
        'trigger': [trigger],
        'touchpad': [touchpad],
        'thumbstick': [touchpad],
        'A': [a],
        'B': [menu],
        'X': [a],
        'Y': [menu],
        'all': [grip, menu, trigger, touchpad, a],
    }
    openvr_buttons = openvr_buttons[button]

    for hc in hclist:
        for button in openvr_buttons:
            if mode == 'default':
                hc.set_default_hand_mode(button)
            else:
                hc.set_hand_mode(button, mode)

# -----------------------------------------------------------------------------
#
def vr_room_camera(session, enable, field_of_view = None, width = None, background_color = None):
    '''
    Mirror using fixed camera in room separate from VR headset view.

    By default VR mirroring shows the right eye view seen in the VR headset.
    This command allows instead using a camera view fixed in room coordinates.

    Parameters
    ----------
    enable : Whether to use a separate room camera for VR mirroring.
    field_of_view : float
      Horizontal field of view of room camera.  Degrees.  Default 90.
    width : float
      Width of room camera screen shown in VR in meters.  Default 1.
    background_color : Color
      Color of background in room camera rendering.  Default is dark gray.
    '''

    c = vr_camera(session)
    rc = c.enable_room_camera(enable)

    if enable:
        if field_of_view is not None:
            rc._field_of_view = field_of_view
        if width is not None:
            rc._camera_model.set_size(width)
        if background_color is not None:
            rc._background_color = background_color.rgba


# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, StringArg, ColorArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('display', EnumOf(('mirror', 'independent', 'blank'))),
                              ('gui', StringArg),
                              ('center', BoolArg),
                              ('click_range', FloatArg),
                              ('multishadow_allowed', BoolArg),
                              ('simplify_graphics', BoolArg),
                   ],
                   synopsis = 'Start SteamVR virtual reality rendering')
    register('vr', desc, vr, logger=logger)
    create_alias('device vr', 'vr $*', logger=logger,
            url="help:user/commands/device.html#vr")

    button_name = EnumOf(('trigger', 'grip', 'touchpad', 'thumbstick', 'menu', 'A', 'B', 'X', 'Y', 'all'))
    desc = CmdDesc(required = [('button', button_name),
                               ('mode', VRModeArg)],
                   keyword = [('hand', EnumOf(('left', 'right')))],
                   synopsis = 'Assign VR hand controller buttons')
    register('vr button', desc, vr_button, logger=logger)
    create_alias('device vr button', 'vr button $*', logger=logger,
            url="help:user/commands/device.html#vr-button")

    desc = CmdDesc(required = [('enable', BoolArg)],
                   keyword = [('field_of_view', FloatArg),
                              ('width', FloatArg),
                              ('background_color', ColorArg)],
                   synopsis = 'Control VR room camera')
    register('vr roomCamera', desc, vr_room_camera, logger=logger)
    create_alias('device vr roomCamera', 'vr roomCamera $*', logger=logger,
            url="help:user/commands/device.html#vr-roomCamera")

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError
class VRModeArg(Annotation):
    '''Command argument for specifying VR hand controller mode.'''

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import EnumOf
        mode_arg = EnumOf(hand_mode_names(session) + ('default',))
        mode_name, used, rest = mode_arg.parse(text, session)
        if mode_name is 'default':
            hm = 'default'
        else:
            c = vr_camera(session)
            hm = c.user_interface._hand_mode_from_name(mode_name)
            if hm is None:
                raise AnnotationError('Unknown VR hand mode "%s"' % mode_name)
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

    c = vr_camera(session)
    if c is session.main_view.camera:
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
        
    try:
        c.start_vr()
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

    session.main_view.camera = c

    # VR gui cannot display a native file dialog.
    session.ui.main_window.use_native_open_dialog = False
    
    # Set redraw timer to redraw as soon as Qt events processsed to minimize dropped frames.
    session.update_loop.set_redraw_interval(0)

    msg = 'started SteamVR rendering'
    log = session.logger
    log.status(msg)
    log.info(msg)
        
# -----------------------------------------------------------------------------
#
def vr_camera(session, create = True):
    c = getattr(session, '_steamvr_camera', None)
    if c is None and create:
        session._steamvr_camera = c = SteamVRCamera(session)
        session.add_state_manager('_steamvr_camera', c)	# For session saving
    return c

# -----------------------------------------------------------------------------
#
def stop_vr(session, simplify_graphics = True):

    c = vr_camera(session, create = False)
    if c is None:
        return

    c.close()
    
    from chimerax.core.graphics import MonoCamera
    v = session.main_view
    v.camera = MonoCamera()
    session.update_loop.set_redraw_interval(10)
    if simplify_graphics:
        from chimerax.std_commands.graphics import graphics_quality
        graphics_quality(session, total_atom_triangles=5000000, total_bond_triangles=5000000)
    from chimerax.label.label3d import label_orient
    label_orient(session, 0)	# Continuously reorient labels.
    v.view_all()
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
from chimerax.core.state import StateManager	# For session saving
class SteamVRCamera(Camera, StateManager):

    always_draw = True	# Draw even if main window iconified.
    
    def __init__(self, session):

        Camera.__init__(self)
        StateManager.__init__(self)

        self._session = session
        self._framebuffers = []		# For rendering each eye view to a texture
        self._texture_drawing = None	# For desktop graphics display
        from sys import platform
        self._use_opengl_flush = (platform == 'darwin')	# On macOS 10.14.1 flickers without glFlush().

        self._hand_controllers = [HandController(self, 'right'),
                                  HandController(self, 'left')]	# List of HandController
        self._controller_show = True	# Whether to show hand controllers
        self._controller_next_id = 0	# Used when searching for controllers.

        self.user_interface = UserInterface(self, session)
        self._vr_model_group = None	# Grouping model for hand controllers and UI models
        self._vr_model_group_id = 100	# Keep VR model group at bottom of model panel

        self._mirror = True		# Whether to render to desktop graphics window.
        self._room_camera = None	# RoomCamera, fixed view camera independent of VR headset

        from chimerax.core.geometry import Place
        self.room_position = Place()	# ChimeraX camera coordinates to room coordinates
        self._room_to_scene = None	# Maps room coordinates to scene coordinates
        self._z_near = 0.1		# Meters, near clip plane distance
        self._z_far = 500.0		# Meters, far clip plane distance
        # TODO: Scaling models to be huge causes clipping at far clip plane.

    def start_vr(self):
        import openvr
        self._vr_system = vrs = openvr.init(openvr.VRApplication_Scene)
        # The init() call raises OpenVRError if SteamVR is not installed.
        # Handle this in the code that tries to create the camera.

        self._render_size = vrs.getRecommendedRenderTargetSize()
        self._compositor = c = openvr.VRCompositor()
        if c is None:
            raise RuntimeError("Unable to create compositor") 

        # Compute projection and eye matrices, units in meters

        # Left and right projections are different. OpenGL 4x4.
        z_near, z_far = self._z_near, self._z_far
        pl = vrs.getProjectionMatrix(openvr.Eye_Left, z_near, z_far)
        self._projection_left = hmd44_to_opengl44(pl)
        pr = vrs.getProjectionMatrix(openvr.Eye_Right, z_near, z_far)
        self._projection_right = hmd44_to_opengl44(pr)

        # Eye shifts from hmd pose.
        vl = vrs.getEyeToHeadTransform(openvr.Eye_Left)
        self._eye_shift_left = hmd34_to_position(vl)
        vr = vrs.getEyeToHeadTransform(openvr.Eye_Right)
        self._eye_shift_right = hmd34_to_position(vr)

        # Map ChimeraX scene coordinates to OpenVR room coordinates
        if self._room_to_scene is None:
            self.fit_scene_to_room()
        
        # Update camera position every frame.
        self._frame_started = False
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self._poses = poses_t()
        t = self._session.triggers
        self._new_frame_handler = t.add_handler('new frame', self.next_frame)

        # Exit cleanly
        self._app_quit_handler = t.add_handler('app quit', self._app_quit)

    @property
    def active(self):
        return self is self._session.main_view.camera
    
    def _move_camera_in_room(self, position):
        '''
        Move camera to the given scene position without changing
        the scene position within the room.  This is done whenever
        the VR headset moves.
        '''
        Camera.set_position(self, position)
        
    def _get_position(self):
        '''VR head mounted display position in the scene.'''
        return Camera.get_position(self)
    def _set_position(self, position):
        '''
        Move camera scene position while keeping it at a fixed position in the room.
        This is for when the mouse moves the camera while in VR.
        '''
        move = position * self.position.inverse()
        Camera.set_position(self, position)
        self.room_to_scene = move * self.room_to_scene
    position = property(_get_position, _set_position)
    
    def _get_room_to_scene(self):
        return self._room_to_scene
    def _set_room_to_scene(self, p):
        self._room_to_scene = p
        # Update positions of models that have fixed room positions.
        self._reposition_user_interface()
        self._reposition_room_camera(p)
    room_to_scene = property(_get_room_to_scene, _set_room_to_scene)
    '''Transformation from room coordinates to scene coordinates.'''
        
    def _reposition_user_interface(self):
        ui = self.user_interface
        if ui.shown():
            ui.move()

    def _reposition_room_camera(self, position):
        rc = self._room_camera
        if rc:
            rc.scene_moved(position)

    def _get_mirror(self):
        return self._mirror
    def _set_mirror(self, enable):
        if enable == self._mirror:
            return
        self._mirror = enable
    mirror = property(_get_mirror, _set_mirror)
    
    def enable_room_camera(self, enable):
        rc = self._room_camera
        if enable and rc is None:
            parent = self._vr_control_model_group()
            self._room_camera = rc = RoomCamera(parent, self.room_to_scene, self.render)
        elif not enable and rc:
            rc.close(self.render)
            self._room_camera = None
        return rc
    
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
            g = self._vr_model_group
            if g is None:
                b = self.vr_view.drawing_bounds()
            else:
                # Need to exclude UI from bounds.
                top_models = self._session.models.scene_root_model.child_models()
                from chimerax.core.geometry import union_bounds
                b = union_bounds(m.bounds() for m in top_models if m.id[0] != g.id[0])
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
            hc.update_scene_position()

    def close(self):

        t = self._session.triggers
        nfh = self._new_frame_handler
        if nfh:
            t.remove_handler(nfh)
            self._new_frame_handler = None

        aqh = self._app_quit_handler
        if aqh:
            t.remove_handler(aqh)
            self._app_quit_handler = None
        
        for hc in self._hand_controllers:
            hc.close()
        
        self.user_interface.close()

        rc = self._room_camera
        if rc:
            rc.close(self.render)
            self._room_camera = None

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
        self._vr_system = None
        self._compositor = None
        self._delete_framebuffers()

        self._session.main_view.redraw_needed = True
    
    def _app_quit(self, tname, tdata):
        # On Linux (Ubuntu 18.04) the ChimeraX process does not exit
        # if VR has not been shutdown.
        import openvr
        openvr.shutdown()

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
        c = self._compositor
        if c is None:
            return
        c.waitGetPoses(renderPoseArray = self._poses, gamePoseArray = None)
        self._frame_started = True

    def device_position(self, device_index):
        dp = self._poses[device_index].mDeviceToAbsoluteTracking
        return hmd34_to_position(dp)
    
    def next_frame(self, *_):
        c = self._compositor
        if c is None:
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
        vrs = self._vr_system
        import openvr
        e = openvr.VREvent_t()
        while vrs.pollNextEvent(e):
            for hc in self.hand_controllers():
                hc.process_event(e)
                
    def process_controller_motion(self):

        for hc in self.hand_controllers():
            hc.process_motion()

    @property
    def desktop_camera_position(self):
        '''Used for moving view with mouse when desktop camera is indpendent of vr camera.'''
        rc = self._room_camera
        return rc.camera_position if rc else None

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        elif view_num == 2:
            v = self._room_camera.camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            es = self._eye_shift_left if view_num == 0 else self._eye_shift_right
            t = es.scale_translation(1/self.scene_scale)
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        draw_desktop = (self._room_camera and self._session.ui.main_window.graphics_window.is_drawable)
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
            p = self._room_camera.projection_matrix(near_far_clip, view_num, window_size)
            return p
        elif view_num == 0:
            p = self._projection_left
        elif view_num == 1:
            p = self._projection_right
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
        elif view_num == 1:  # VR right-eye
            # Submit left eye texture (view 0) before rendering right eye (view 1)
            self._submit_eye_image('left', left_fb.openvr_texture, render)
            render.pop_framebuffer()
            render.push_framebuffer(right_fb)
        elif view_num == 2: # independent camera desktop view
            # Submit right eye texture (view 1) before rendering desktop (view 2)
            self._submit_eye_image('right', right_fb.openvr_texture, render)
            render.pop_framebuffer()
            self._room_camera.start_rendering(render)

    def _submit_eye_image(self, side, texture, render):
        '''Side is "left" or "right".'''
        import openvr
        eye = openvr.Eye_Left if side == 'left' else openvr.Eye_Right
        # Caution: compositor.submit() changes the OpenGL read framebuffer binding to 0.
        result = self._compositor.submit(eye, texture)
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
        if self.number_of_views() == 2:
            rtex = render.current_framebuffer().openvr_texture
            self._submit_eye_image('right', rtex, render)

        render.pop_framebuffer()
        
        if self.mirror:
            # Render right eye to ChimeraX window.
            drawing = self._desktop_drawing()
            from chimerax.core.graphics.drawing import draw_overlays
            draw_overlays([drawing], render)

        rc = self._room_camera
        if rc:
            rc.finish_rendering(render)
            
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

    def _desktop_drawing(self):
        '''Used  to render ChimeraX desktop graphics window.'''
        rc = self._room_camera
        if rc:
            texture = rc.framebuffer(self.render).color_texture
        else:
            texture = self._framebuffers[1].color_texture
        td = self._texture_drawing
        if td is None:
            # Drawing object for rendering to ChimeraX window
            from chimerax.core.graphics.drawing import _texture_drawing
            self._texture_drawing = td = _texture_drawing(texture)
            td.opaque_texture = True
        else:
            td.texture = texture
        window_size = self.render.render_size()
        from chimerax.core.graphics.drawing import match_aspect_ratio
        match_aspect_ratio(td, window_size)
        return td

    def do_swap_buffers(self):
        return self.mirror

    def hand_controllers(self):
        if not self._all_hand_controllers_on:
            self._find_new_hand_controllers()
        return self._hand_controllers

    @property
    def _all_hand_controllers_on(self):
        for hc in self._hand_controllers:
            hc.check_if_model_closed()
            if not hc.on:
                return False
        return True

    def _find_new_hand_controllers(self):
        # Check if a controller has been turned on.
        # Only check one controller id per-call to minimize performance penalty.
        import openvr
        d = self._controller_next_id
        self._controller_next_id = (d+1) % openvr.k_unMaxTrackedDeviceCount
        vrs = self._vr_system
        if (vrs.getTrackedDeviceClass(d) == openvr.TrackedDeviceClass_Controller
            and vrs.isTrackedDeviceConnected(d)):
            left_or_right = self._controller_left_or_right(d)
            for hc in self._hand_controllers:
                if hc.left_or_right == left_or_right:
                    hc.set_device_index(d)

    def _controller_left_or_right(self, device_index):
        vrs = self._vr_system
        import openvr
        left_id = vrs.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_LeftHand)
        return 'left' if device_index == left_id else 'right'

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

    # Session save.
    def take_snapshot(self, session, flags):
        data = {'room_to_scene': self.room_to_scene,
                'button_assignments': tuple(hc.button_assignments for hc in self._hand_controllers),
                'active': self.active,
                'version': 1
                }
        return data

    # Session restore.
    @classmethod
    def restore_snapshot(cls, session, data):
        """Create object using snapshot data."""
        c = vr_camera(session)
        c.room_to_scene = data['room_to_scene']
        for hc, ba in zip(c._hand_controllers, data['button_assignments']):
            hc.button_assignments = ba
        if data['active']:
            # Try to start VR if it was active when session saved.
            def start_vr(trigger_name, session):
                try:
                    vr(session, enable = True)
                except Exception as e:
                    # Failed to start VR.
                    session.logger.info(str(e))
                from chimerax.core.triggerset import DEREGISTER
                return DEREGISTER
            session.triggers.add_handler('end restore session', start_vr)
        return c

    def reset_state(self, session):
        pass
    
class RoomCamera:
    '''Camera fixed in room for mirroring to desktop.'''
    def __init__(self, parent, room_to_scene, render):
        self._framebuffer = None	# Framebuffer for rendering room camera view.
        self._camera_model = None
        self._field_of_view = 90	# Degrees.  Horizontal.
        self._background_color = (.1,.1,.1,1)	# RGBA, float 0-1
        self._settings = None		# Saved preferences, room position.

        # Depiction of camera in VR scene.
        render.make_current()	# Texture is allocated when framebuffer created.
        texture = self.framebuffer(render).color_texture
        self._camera_model = self._create_camera_model(parent, room_to_scene, texture)

    def delete(self, render):
        self._delete_framebuffer(render)

    def close(self, render):
        self._delete_framebuffer(render)
        self._save_camera_position()
        cm = self._camera_model
        if cm:
            cm.delete()
            self._camera_model = None
        
    @property
    def enabled(self):
        return self._camera_model is not None

    @property
    def camera_position(self):
        cm = self._camera_model
        if cm is None:
            from chimerax.core.geometry import Place
            p = Place()
        else:
            p = cm.position
        return p

    def scene_moved(self, new_room_to_scene):
        '''
        Adjust camera scene position so that it stays
        at the same position in the room.
        '''
        cm = self._camera_model
        if cm:
            cm.update_scene_position(new_room_to_scene)

    def projection_matrix(self, near_far_clip, view_num, window_size):
        pixel_shift = (0,0)
        fov = self._field_of_view
        from chimerax.core.graphics.camera import perspective_projection_matrix
        return perspective_projection_matrix(fov, window_size, near_far_clip, pixel_shift)
    
    def _create_camera_model(self, parent, room_to_scene, texture):
        cm = RoomCameraModel('Room camera', parent.session, texture, room_to_scene)
        cm.room_position = self._initial_room_position(parent.session)
        parent.add([cm])
        return cm

    def _initial_room_position(self, session):
        if self._settings is None:
            from chimerax.core.geometry import translation
            # Centered 1.5 meters off floor, 2 meters from center
            default_position = translation((0, 1.5, 2))
            m = tuple(tuple(row) for row in default_position.matrix)
            from chimerax.core.settings import Settings
            class _VRRoomCameraSettings(Settings):
                EXPLICIT_SAVE = {
                    'independent_camera_position': m,
                }
            self._settings = _VRRoomCameraSettings(session, "vr_room_camera")
        from chimerax.core.geometry import Place
        p = Place(self._settings.independent_camera_position)
        return p

    def _save_camera_position(self):
        settings = self._settings
        cm = self._camera_model
        if settings is None or cm is None:
            return
        m = tuple(tuple(row) for row in cm.room_position.matrix)
        settings.independent_camera_position = m
        settings.save()
        
    def start_rendering(self, render):
        fb = self.framebuffer(render)
        render.push_framebuffer(fb)

        # Set paramters for mixed reality blending.
        render.mix_video = True  # For making mixed reality videos

        # Don't render camera model in desktop camera view.
        self.enable_draw = False
        
        # Make background contrast with room background so vr user can see boundary.
        render.set_background_color(self._background_color)

    def finish_rendering(self, render):
        # Turn off mixed reality blending.
        render.mix_video = False

        # Reenable camera model rendering for VR eye views.
        self.enable_draw = True

    def enable_draw(self, enable):
        cm = self._camera_model
        if cm:
            cm.enable_draw = enable

    def framebuffer(self, render):
        rfb = render.default_framebuffer()
        tw,th = rfb.width, rfb.height
        fb = self._framebuffer
        if fb is None or fb.width != tw or fb.height != th:
            self._delete_framebuffer(render)
            from chimerax.core.graphics import Texture, opengl
            t = Texture()
            t.initialize_rgba((tw,th))
            fb = opengl.Framebuffer('VR desktop', render.opengl_context, color_texture = t)
            self._framebuffer = fb
            cm = self._camera_model
            if cm:
                cm.texture = t
                cm.set_size()	# Adjust for new aspect ratio.
        return fb

    def _delete_framebuffer(self, render):
        fb = self._framebuffer
        if fb:
            render.make_current()
            fb.delete()
            self._framebuffer = None
    
from chimerax.core.models import Model
class RoomCameraModel(Model):
    '''
    Depict camera in scene when fixed position camera is used
    to render the desktop graphics window.  The camera looks in the -z direction.
    The camera is shown as a rectangle and texture mapped onto it is what the camera sees.
    '''
    casts_shadows = False
#    skip_bounds = True   # Camera screen disappears if it is far from models
    SESSION_SAVE = False

    def __init__(self, name, session, texture, room_to_scene, width = 1):
        '''Width in meters.'''
        self.enable_draw = True
        self._last_room_to_scene = room_to_scene
        self._width = width

        Model.__init__(self, name, session)

        self.color = (255,255,255,255)	# Don't modulate texture colors.
        self.use_lighting = False
        self.texture = texture
        self.opaque_texture = True
        self.set_size(width)

        # Avoid camera disappearing when far from models
        self.allow_depth_cue = False

    def _get_room_position(self):
        return (self._last_room_to_scene.inverse() * self.position).remove_scale()
    def _set_room_position(self, room_position):
        self.position = (self._last_room_to_scene * room_position).remove_scale()
    room_position = property(_get_room_position, _set_room_position)

    def set_size(self, width=None):
        if width is None:
            width = self._width
        else:
            self._width = width
        scene_width = width * self._last_room_to_scene.scale_factor()
        tw, th = self.texture.size
        scene_height = th * scene_width/tw
        va, na, tc, ta = self._geometry(scene_width, scene_height)
        self.set_geometry(va, na, ta)
        self.texture_coordinates = tc
    
    def _geometry(self, width, height):
        '''Depict camera as a rectangle perpendicular to z axis.'''
        w, h = .5 * width, .5 * height
        from numpy import array, float32, int32, uint8
        vertices = array([(-w,-h,0),(-w,h,0),(w,h,0),(w,-h,0)], float32)
        normals = array([(0,0,-1),(0,0,-1),(0,0,-1),(0,0,-1)], float32)
        texcoords = array([(1,0),(1,1),(0,1),(0,0)], float32)
        triangles = array([(0,1,2),(0,2,3)], int32)
        return vertices, normals, texcoords, triangles

    def draw(self, renderer, draw_pass):
        if self.enable_draw:
            Model.draw(self, renderer, draw_pass)
            
    def update_scene_position(self, new_rts):
        old_rts = self._last_room_to_scene
        self._last_room_to_scene = new_rts
        move = new_rts * old_rts.inverse()
        mpos = move * self.position
        # Need to remove scale factor.
        from chimerax.core.geometry import norm, Place
        s = norm(move.matrix[:,0])
        m = mpos.matrix
        m[:3,:3] *= 1/s
        self.position = Place(m)
        if abs(s - 1) > 1e-5:
            # Keep camera same size in room coordinates.
            self.set_geometry(s*self.vertices, self.normals, self.triangles)

class UserInterface:
    '''
    Panel in VR showing ChimeraX main window.
    Buttons can be clicked with hand controllers.
    '''
    def __init__(self, camera, session):
        self._camera = camera
        self._session = session

        self._mouse_mode_click_range = 5 # In scene units (Angstroms).
        self._update_later = 0		# Redraw panel after this many frames
        self._update_delay = 10		# After click on panel, update after this number of frames
        self._ui_model = None
        self._panels = []		# List of Panel, one for each user interface pane
        self._gui_tool_names = None	# List of ToolInstance names to show panels for.  None shows all visible tools.
        self._panel_y_spacing = 0.01	# meters
        self._panel_z_spacing = 0.001	# meters
        self._buttons_down = {}		# (HandController, button) -> Panel
        self._raised_buttons = {}	# maps highlight_id to (widget, panel)
        self._move_gui = set()		# set of (HandController, button) if gui being moved by press on title bar
        self._move_ui_mode = MoveUIMode()
        self._tool_show_handler = None

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

        h = self._tool_show_handler
        if h:
            triggers = self._session.ui.triggers
            triggers.remove_handler(h)
            self._tool_show_handler = None
            
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

        # Menu bar
        if self._gui_tool_names is None:
            menu_bar = self._session.ui.main_window.menuBar()
            p = Panel(menu_bar, ui, self, tool_name = 'menu bar')
            panels.append(p)

        # Tools
        exclude_tools = set(['Command Line Interface'])
        tool_names = self._gui_tool_names
        if tool_names is None:
            # Show all displayed tools.
            tools = [ti for ti in self._session.tools.list()
                     if hasattr(ti, 'tool_window') and ti.displayed()
                        and ti.tool_name not in exclude_tools]
            tools.sort(key = _tool_y_position)
            tool_names = [ti.tool_name for ti in tools]
        for tool_name in tool_names:
            w = _tool_widget(tool_name, self._session)
            if w:
                p = Panel(w, ui, self, tool_name = tool_name)
                panels.append(p)
            else:
                self._session.logger.warning('VR user interface could not find tool "%s"' % tool_name)

        # Position panels on top of each other
        self._stack_panels(panels)

        # Add panels for non-tools like recent files panel.
        self._check_for_new_panels()
        
        # Monitor when windows are shown and hidden.
        triggers = self._session.ui.triggers
        self._tool_show_handler = triggers.add_handler('tool window show or hide',
                                                       self._tool_window_show_or_hide)
        return panels

    def _stack_panels(self, panels):
        sep = self._panel_y_spacing
        dz = self._panel_z_spacing
        spanels = [p for p in panels if not p.is_menu()]
        h = sum(p.size[1] for p in spanels) + (len(spanels)-1)*sep
        # Stack panels.
        y = h/2
        z = -dz
        from chimerax.core.geometry import translation
        for p in spanels:
            h = p.size[1]
            y -= 0.5*h
            pd = p._panel_drawing
            pd.position = translation((0,y,z))
            y -= 0.5*h + sep
            z -= dz

        if len(panels) > len(spanels):
            # Position menu panels.
            mpanels = [p for p in panels if p.is_menu()]
            for mp in mpanels:
                mp.position_menu_over_parent(spanels)
                
    def _tool_window_show_or_hide(self, trig_name, tool_window):
        if tool_window.shown:
            self._add_tool_panel(tool_window)
        else:
            self._delete_tool_panel(tool_window)

    def _add_tool_panel(self, tool_window):
        tool_name = tool_window.tool_instance.tool_name
        if self._find_tool_panel(tool_name):
            return
        w = _tool_widget(tool_name, self._session)
        if w is None:
            return
        p = Panel(w, self._ui_model, self, tool_name = tool_name)
        self._panels.append(p)
        self.redraw_ui()

    def _check_for_new_panels(self):
        # Add new panels for newly appeared top level widgets.
        from PyQt5.QtWidgets import QDockWidget, QMainWindow, QMenu
        tw = [w for w in self._session.ui.topLevelWidgets()
              if w.isVisible() and not isinstance(w, (QDockWidget, QMainWindow))]
        wset = set(p._widget for p in self._panels)
        neww = [w for w in tw if w not in wset]
        newp = [Panel(w, self._ui_model, self, tool_name = w.windowTitle()) for w in neww]
        self._panels.extend(newp)
        
        for p in newp:
            if p.is_menu():
                p.position_menu_over_parent(self._panels)

        # Show rapid access panel
        w = self._session.ui.main_window.rapid_access
        if w.isVisible() and w not in wset:
            p = Panel(w, self._ui_model, self, tool_name = 'Recent Files',
                      add_titlebar = True)
            self._panels.append(p)
            
        if neww:
            self.redraw_ui()

        # Remove closed panels
        for p in tuple(self._panels):
            try:
                vis = p._widget.isVisible()
            except:
                vis = False	# Panel destroyed
            if not vis:
                self._delete_panel(p)

    def _find_tool_panel(self, tool_name):
        for p in self._panels:
            if p.name == tool_name:
                return p
        return None

    def _delete_tool_panel(self, tool_window):
        tool_name = tool_window.tool_instance.tool_name
        p = self._find_tool_panel(tool_name)
        if p:
            self._delete_panel(p)
        self.redraw_ui()

    def _close_menu_panels(self):
        # Menus do not automatically close when a VR generated mouse event
        # is posted on Windows 10.  It seems to take a real mouse click to dismiss menus.
        # So this routine explicitly dismisses menus when VR click is made.
        for p in tuple(self._panels):
            if p.is_menu():
                w = p.widget
                if w:
                    w.close()
                    
    def _delete_panel(self, panel):
        self._panels.remove(panel)
        panel.delete(self._ui_model)
        # Forget raised buttons in this panel.
        hids = []
        rb = self._raised_buttons
        for highlight_id, (w, p) in rb.items():
            if p == panel:
                hids.append(highlight_id)
        for hid in hids:
            del rb[hid]
        
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
        '''
        Returns true if button event was on UI panel, otherwise
        false indicating hand controller assigned button mode should be used.
        '''
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
                if window_xy is not None:
                    panel.release(window_xy)
                del bdown[(hc,b)]
                return True
            elif (hc,b) in self._move_gui:
                self._move_gui.remove((hc,b))
                hc._drag_end(self._move_ui_mode)
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
                if panel.clicked_on_close_button(window_xy):
                    self._delete_panel(panel)
                    panel.widget.close()
                elif panel.clicked_on_title_bar(window_xy):
                    # Drag on title bar moves VR gui
                    self._move_gui.add((hc,b))
                    mum = self._move_ui_mode
                    mum.set_panel(panel)
                    hc._drag_start(mum, b)
                else:
                    hand_mode = panel.clicked_mouse_mode(window_xy)
                    if hand_mode is not None:
                        self._enable_mouse_mode(hand_mode, hc, b, window_xy, panel)
                    else:
                        panel.press(window_xy)
                        bdown[(hc,b)] = panel
                if not panel.is_menu():
                    # Menus don't close on VR click without this call.
                    self._close_menu_panels()
                return True

        return False

    def _enable_mouse_mode(self, hand_mode, hand_controller, button, window_xy, panel):
        if isinstance(hand_mode, MouseMode) and not hand_mode.has_vr_support:
            msg = 'No VR support for mouse mode %s' % hand_mode.name
        else:
            hand_controller.set_hand_mode(button, hand_mode)
            msg = 'VR mode %s' % hand_mode.name
        self._session.logger.info(msg)
        panel._show_pressed(window_xy)
        self.redraw_ui()	# Show log message

    def process_hand_controller_motion(self, hand_controller):
        hc = hand_controller
        dragged = False
        for (bhc, b), panel in self._buttons_down.items():
            if hc == bhc:
                window_xy, z_offset = panel._panel_click_position(hc.room_position.origin())
                if window_xy is not None:
                    panel.drag(window_xy)
                    dragged = True
        if dragged:
            return True

        # Highlight ui button under pointer
        self._highlight_button(hc.room_position.origin(), hc)

        return False
            
    def _highlight_button(self, room_point, highlight_id):
        window_xy, panel = self._click_position(room_point)
        if panel:
            widget, wpos = panel.clicked_widget(window_xy)
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
                self._check_for_new_panels()
                self._update_ui_images()

    def _update_ui_images(self):
        ui = self._session.ui
        im = ui.window_image()
        from chimerax.core.graphics.drawing import qimage_to_numpy
        rgba = qimage_to_numpy(im)
        for panel in tuple(self._panels):
            if panel._window_closed():
                self._delete_panel(panel)
            else:
                panel._update_image(rgba)
#            self._stack_panels(self._panels)
        
    def set_mouse_mode_click_range(self, range):
        self._mouse_mode_click_range = range

    def _hand_mode_from_name(self, name, mouse_mode = None):
        mode = hand_mode_by_name(name)
        if mode:
            m = mode()
        else:
            if mouse_mode is None:
                mouse_mode = self._session.ui.mouse_modes.named_mode(name)
            if mouse_mode:
                m = MouseMode(mouse_mode, self._mouse_mode_click_range)
            else:
                m = None
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
        m.color = (255,255,255,255)
        m.use_lighting = False
        # m.skip_bounds = True  # User interface clipped if far from models.
        m.casts_shadows = False
        m.pickable = False
        m.SESSION_SAVE = False
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
    def __init__(self, qt_widget, drawing_parent, ui,
                 tool_name = None, pixel_size = 0.001, add_titlebar = False):
        self._widget = qt_widget	# This Qt widget is shown in the VR panel.
        self._ui = ui			# UserInterface instance
        self._tool_name = tool_name	# Name of tool instance
        th = 20 if add_titlebar or self._needs_titlebar() else 0
        self._titlebar_height = th      # Added titlebar height in pixels
        w,h = self._panel_size
        self._size = (pixel_size*w, pixel_size*h) # Billboard width, height in room coords, meters.
        self._pixel_size = pixel_size	# In meters.

        self._last_image_rgba = None
        self._ui_click_range = 0.05 	# Maximum distance of click from plane, room coords, meters.
        self._button_rise = 0.01	# meters rise when pointer over button
        self._panel_thickness = 0.01	# meters

        # Drawing that renders this panel.
        self._panel_drawing = self._create_panel_drawing(drawing_parent)

    @property
    def widget(self):
        w = self._widget
        try:
            w.width()
        except:
            w = None	# Widget was deleted.
        return w

    def _create_panel_drawing(self, drawing_parent):
        from chimerax.core.graphics import Drawing
        d = Drawing('VR UI panel')
        d.color = (255,255,255,255)
        d.use_lighting = False
        # d.skip_bounds = True	# Clips if far from models.
        drawing_parent.add_drawing(d)
        return d

    def delete(self, parent):
        pd = self._panel_drawing
        if pd:
            parent.remove_drawings([pd])
            self._panel_drawing = None
            
    @property
    def name(self):
        n = self._tool_name
        return 'unnamed gui panel' if n is None else n

    @property
    def size(self):
        '''Panel width and height in room coordinate system (meters).'''
        return self._size

    @property
    def drawing(self):
        return self._panel_drawing

    def move(self, room_motion):
        pd = self._panel_drawing
        room_to_scene = self._ui._camera.room_to_scene
        room_pos = room_to_scene.inverse() * pd.scene_position
        new_room_pos = room_motion * room_pos
        pd.scene_position = room_to_scene * new_room_pos
        
    def scale_panel(self, scale_factor, center = None):
        '''
        Center is specified in the parent model coordinate system.
        If center is not specified then panel scales about its geometric center.
        '''
        w,h = self.size
        self._size = (scale_factor*w, scale_factor*h)
        self._pixel_size *= scale_factor

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
        if sx is None or sy is None:
            return None, None
        ws = 1/w if w > 0 else 0
        hs = 1/h if h > 0 else 0
        th = self._titlebar_height
        window_xy = sx * (x + hw) * ws, sy * (hh - y) * hs - th
        return window_xy, z_offset

    def _update_image(self, main_window_rgba):
        rgba = self._panel_image(main_window_rgba)
        if rgba is None:
            return False
        lrgba = self._last_image_rgba
        self._last_image_rgba = rgba
        if lrgba is None or rgba.shape != lrgba.shape:
            w,h = self._panel_size
            ps = self._pixel_size
            self._size = (ps*w,ps*h)
            self._update_geometry()

        d = self._panel_drawing
        if d.texture is not None:
            d.texture.reload_texture(rgba)
        else:
            from chimerax.core.graphics import Texture
            d.texture = Texture(rgba)

        return True
    
    def _update_geometry(self):
        # Vertex coordinates are in room coordinates (meters), and
        # position matrix contains scale factor to produce scene coordinates.

        # Calculate rectangles for panel and raised buttons
        w, h = self.size
        xmin,ymin,xmax,ymax = -0.5*w,-0.5*h,0.5*w,0.5*h
        th = self._panel_thickness
        rects = [(xmin,ymin,-th,xmax,ymax,0)]
        zr = self._button_rise
        rb = self._ui._raised_buttons
        for widget, panel in rb.values():
            if panel is self:
                r = self._button_rectangle(widget)
                if r is None:
                    continue
                x0,y0,x1,y1 = r
                z = .5*zr if getattr(widget, '_show_pressed', False) else zr
                rects.append((x0,y0,z-th,x1,y1,z))

        # Create geometry for rectangles
        nr = len(rects)
        nv = 12*nr
        nt = 12*nr
        from numpy import empty, float32, int32
        v = empty((nv,3), float32)
        tc = empty((nv,2), float32)
        t = empty((nt,3), int32)
        ws = 1/w if w > 0 else 0
        hs = 1/h if h > 0 else 0
        for r, (x0,y0,z0,x1,y1,z1) in enumerate(rects):
            ov, ot = 12*r, 12*r
            v[ov:ov+12] = ((x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1), # Front
                           (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1), # Sides and back
                           (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)) # Sides and back
            tx0, ty0, tx1, ty1 = (x0-xmin)*ws, (y0-ymin)*hs, (x1-xmin)*ws, (y1-ymin)*hs
            tc[ov:ov+12] = ((tx0,ty0), (tx1,ty0), (tx1,ty1), (tx0,ty1), # Front
                            (tx0,ty0), (tx0,ty0), (tx0,ty0), (tx0,ty0), # Sides and back
                            (tx0,ty0), (tx0,ty0), (tx0,ty0), (tx0,ty0)) # Sides and back
            faces = [(ov+i,ov+j,ov+k) for i,j,k in ((0,1,2),(0,2,3),(4,8,9),(4,9,5),
                                                    (5,9,10),(5,10,6),(6,10,11),(6,11,7),
                                                    (7,11,8),(7,8,4),(8,11,10),(8,10,9))]
            t[ot:ot+12] = faces

        # Update Drawing
        d = self._panel_drawing
        d.set_geometry(v, None, t)
        d.texture_coordinates = tc

    def panel_image_rgba(self):
        return self._last_image_rgba

    def _panel_image(self, main_window_rgba):
        rgba = self._widget_rgba()
        return rgba

    def _widget_rgba(self):
        w = self.widget
        if w is None:
            return None
        # TODO: grab() does not include the Windows title bar in the image returned.
        #  We want the title bar because it gives the name of the tool.
        #  Looks like Qt can't get the title bar.  I may want to add a title to the
        #  top of the grabbed image.
        pixmap = w.grab()
        im = pixmap.toImage()
        from chimerax.core.graphics.drawing import qimage_to_numpy
        rgba = qimage_to_numpy(im)
        trgba = self._add_titlebar(rgba)
        return trgba

    def _add_titlebar(self, rgba, title_color = (0,0,0,255), background_color = (210,210,210,255)):
        th = self._titlebar_height
        if th == 0:
            return rgba
        
        h,ww,c = rgba.shape
        from numpy import empty
        trgba = empty((h+th,ww,c), rgba.dtype)
        trgba[:h,:,:] = rgba

        # Add title text
        trgba[h:,:,:] = background_color
        title = self.name
        if title:
            from chimerax.core.graphics import text_image_rgba
            title_rgba = text_image_rgba(title, title_color, th, 'Arial',
                                         background_color = background_color,
                                         xpad = 8, ypad = 4, pixels = True)
            tw = min(title_rgba.shape[1], trgba.shape[1])
            trgba[h:,:tw,:] = title_rgba[:,:tw,:]

        # Add close button
        x_sign = '\u00D7'	# Unicode multiply symbol
        from chimerax.core.graphics import text_image_rgba
        x_rgba = text_image_rgba(x_sign, title_color, th, 'Arial',
                                 background_color = background_color,
                                 xpad = 6, pixels = True)
        xw = min(x_rgba.shape[1], trgba.shape[1])
        trgba[h:,-xw:,:] = x_rgba[:,:xw,:]

        return trgba

    def is_menu(self):
        from PyQt5.QtWidgets import QMenu
        return isinstance(self.widget, QMenu)
    
    def _is_toplevel_widget(self):
        w = self.widget
        if w is None:
            return False
        top = w.window()
        return w == top

    def _needs_titlebar(self):
        return self._is_toplevel_widget() and not self.is_menu()
    
    @property
    def _panel_size(self):
        '''In pixels.'''
        pw = self.widget
        if pw is None:
            return None, None
        return pw.width(),pw.height() + self._titlebar_height

    def _button_rectangle(self, widget):
        '''Returns coordinates in meters with 0,0 at center of ui panel.'''
        w, h = self._panel_size
        if w is None:
            return None
        xc, yc = 0.5*w, 0.5*h
        pw = self.widget
        if pw is None:
            return None
        from PyQt5.QtCore import QPoint
        wxy0 = widget.mapTo(pw, QPoint(0,0))
        th = self._titlebar_height
        wx0,wy0 = wxy0.x(), wxy0.y() + th
        ww,wh = widget.width(), widget.height()
        wx1, wy1 = wx0+ww, wy0+wh
        pw, ph = self.size
        ws = 1/w if w > 0 else 0
        hs = 1/h if h > 0 else 0
        rect = (pw*(wx0-xc)*ws, -ph*(wy0-yc)*hs, pw*(wx1-xc)*ws, -ph*(wy1-yc)*hs)
        return rect

    def _window_closed(self):
        w = self.widget
        if w is None or not w.isVisible():
            return True
        return False

    def press(self, window_xy):
        return self._click('press', window_xy)

    def drag(self, window_xy):
        return self._click('move', window_xy)

    def release(self, window_xy):
        return self._click('release', window_xy)

    def _click(self, type, window_xy):
        '''Type can be "press" or "release".'''
        w = self._post_mouse_event(type, window_xy)
        if w:
            if type == 'press':
                self._show_pressed_button(w)
            if type == 'release':
                self._show_pressed_button(w, pressed = False)
                self._ui.redraw_ui()
            return True
        return False
    
    def _post_mouse_event(self, type, window_xy):
        '''Type is "press", "release" or "move".'''
        w, pos = self.clicked_widget(window_xy)
        if w is None or pos is None:
            return w
        from PyQt5.QtCore import Qt, QEvent
        if type == 'press':
            from time import time
            t = time()
            double_click = (hasattr(self, '_last_click_time')
                            and t - self._last_click_time < 0.5)
            et = QEvent.MouseButtonDblClick if double_click else QEvent.MouseButtonPress
            self._last_click_time = t
            button = buttons = Qt.LeftButton
        elif type == 'release':
            et = QEvent.MouseButtonRelease
            button = Qt.LeftButton
            buttons =  Qt.NoButton
        elif type == 'move':
            et = QEvent.MouseMove
            button =  Qt.NoButton
            buttons = Qt.LeftButton
        from PyQt5.QtGui import QMouseEvent
        me = QMouseEvent(et, pos, button, buttons, Qt.NoModifier)
        self._ui._session.ui.postEvent(w, me)
        return w

    def clicked_widget(self, window_xy):
        # Input window_xy coordinates are in top level window that panel is in.
        # Returns clicked widget and (x,y) in widget pixel coordinate system.
        pw = self.widget
        if pw is None:
            return None, None
        from PyQt5.QtCore import QPoint, QPointF
        x,y = window_xy
        pwp = QPoint(int(x), int(y))
        w = pw.childAt(pwp)	# Works even if widget is covered.
        if w is None:
            return pw, pwp
        gp = pw.mapToGlobal(pwp)
        # Using w = ui.widgetAt(gp) does not work if the widget is covered by another app.
        wpos = QPointF(w.mapFromGlobal(gp)) if w else None
        return w, wpos

    def _show_pressed(self, window_xy, pressed = True):
        w, wpos = self.clicked_widget(window_xy)
        if w:
            self._show_pressed_button(w, pressed)

    def _show_pressed_button(self, widget, pressed = True):
        rb = self._ui._raised_buttons
        for w, panel in rb.values():
            if w == widget:
                widget._show_pressed = pressed
                self._update_geometry()	# Show partially depressed button

    def clicked_on_close_button(self, window_xy):
        th = self._titlebar_height
        if th > 0:
            x,y = window_xy
            return y < 0 and x >= self._panel_size[0]-th
        return False
        
    def clicked_on_title_bar(self, window_xy):
        th = self._titlebar_height
        if th > 0:
            return window_xy[1] < 0
        w, pos = self.clicked_widget(window_xy)
        from PyQt5.QtWidgets import QMenuBar, QDockWidget
        if isinstance(w, QMenuBar) and w.actionAt(pos) is None:
            return True
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        return isinstance(w, (QDockWidget, TabbedToolbar))
                           
    def clicked_mouse_mode(self, window_xy):
        w, pos = self.clicked_widget(window_xy)
        from PyQt5.QtWidgets import QToolButton
        if isinstance(w, QToolButton):
            if hasattr(w, 'vr_mode'):
                if isinstance(w.vr_mode, str):
                    mouse_mode = self._ui._session.ui.mouse_modes.named_mode(w.vr_mode)
                else:
                    mouse_mode = w.vr_mode()
                return self._ui._hand_mode_from_name(mouse_mode.name, mouse_mode)
            a = w.defaultAction()
            if hasattr(a, 'vr_mode'):
                mouse_mode = a.vr_mode()
                return self._ui._hand_mode_from_name(mouse_mode.name, mouse_mode)
        return None

    def position_menu_over_parent(self, panels):
        # Try to use parent widget to find panel to align with.
        p = self._parent_panel(panels)
        if p is None:
            # This can happen of QMenu() was made without specifying a parent.
            return	# Menu is not child of any panels.

        w = self.widget
        if w is None:
            return
        
        from PyQt5.QtCore import QPoint
        pos = w.mapToGlobal(QPoint(0,0))
        ppos = p._widget.mapFromGlobal(pos)
        ps = p._pixel_size
        pw,ph = p._size
        sw,sh = self._size
        offset = (ppos.x()*ps + sw/2 - pw/2, -(ppos.y()*ps + sh/2 - ph/2), .01)
        pd = self._panel_drawing
        from chimerax.core.geometry import translation
        pd.position = p._panel_drawing.position * translation(offset)
        
    def _parent_panel(self, panels):
        w = self.widget
        if w is None:
            return None
        a = set(_ancestor_widgets(w))
        for p in panels:
            if p._widget in a:
                return p
        return None

def _ancestor_widgets(w):
    alist = []
    p = w
    while True:
        p = p.parentWidget()
        if p is None:
            return alist
        alist.append(p)

def _tool_y_position(tool_instance):
    if hasattr(tool_instance, 'tool_window'):
        return tool_instance.tool_window._dock_widget.y()
    return None

def _find_tool_by_name(name, session):
    for ti in session.tools.list():
        if ti.tool_name == name:
            return ti
    return None

def _tool_widget(name, session):
    ti = _find_tool_by_name(name, session)
    if ti and hasattr(ti, 'tool_window'):
        w = ti.tool_window._dock_widget
    else:
        w = None
    return w
        
class HandController:
    _controller_colors = {'left':(200,200,0,255), 'right':(0,200,200,255), 'default':(180,180,180,255)}

    def __init__(self, camera, left_or_right = 'right', length = 0.20, radius = 0.04):

        self._camera = camera
        self._side = left_or_right
        self._length = length
        self._radius = radius

        self._device_index = None
        self._hand_model = None
        
        # Assign actions bound to controller buttons
        self._modes = {}		# Maps button name to HandMode
        self._active_drag_modes = set() # Modes with an active drag (ie. button down and not yet released).

        # Settings for Oculus thumbstick used as single-step control
        self._thumbstick_released = False
        self._thumbstick_release_level = 0.2
        self._thumbstick_click_level = 0.5
        self._thumbstick_repeating = False
        self._thumbstick_repeat_delay = 0.3  # seconds
        self._thumbstick_repeat_interval = 0.1  # seconds
        self._thumbstick_time = None

    @property
    def on(self):
        return self._device_index is not None
        
    def set_device_index(self, device_index):
        if device_index == self._device_index:
            return
        self._device_index = device_index
        self._set_controller_type()
        if self._hand_model is None:
            self._create_hand_model()
        self._set_initial_button_assignments()

    @property
    def _vr_system(self):
        return self._camera._vr_system

    def _set_controller_type(self):
        vrs = self._camera._vr_system
        from openvr import Prop_RenderModelName_String
        model_name = vrs.getStringTrackedDeviceProperty(self._device_index,
                                                        Prop_RenderModelName_String)
        # 'vr_controller_vive_1_5' for vive pro
        # 'oculus_cv1_controller_right', 'oculus_cv1_controller_left'
        # 'oculus_rifts_controller_right', 'oculus_rifts_controller_left'
        self._controller_type = model_name

        self._is_oculus = model_name.startswith('oculus')

    def _initial_button_modes(self):

        if not hasattr(self, '_is_oculus'):
            return {}	# VR not started yet, so we don't know controller type.
        
        from openvr import \
            k_EButton_Grip as grip, \
            k_EButton_ApplicationMenu as menu, \
            k_EButton_SteamVR_Trigger as trigger, \
            k_EButton_SteamVR_Touchpad as touchpad, \
            k_EButton_A as a
        
        if self._is_oculus:
            # Oculus touch controller left and right buttons:
            #    trigger = k_EButton_Axis1 = 33 = k_EButton_SteamVR_Trigger
            #    grip = k_EButton_Grip = 2 and k_EButton_Axis2 = 34 both
            #    A or X button = k_EButton_A = 7
            #    B or Y button = k_EButton_ApplicationMenu = 1
            #    thumbstick = k_EButton_Axis0 = 32 = k_EButton_SteamVR_Touchpad
            thumbstick_mode = ZoomMode() if self.left_or_right == 'right' else MoveSceneMode()
            initial_modes = {
                menu: ShowUIMode(),
                trigger: MoveSceneMode(),
                grip: MoveSceneMode(),
                a: ZoomMode(),
                touchpad: thumbstick_mode
            }
        else:
            initial_modes = {
                menu: ShowUIMode(),
                trigger: MoveSceneMode(),
                grip: RecenterMode(),
                touchpad: ZoomMode()
            }

        return initial_modes
    
    def _set_initial_button_assignments(self):
        im = self._initial_button_modes()
        for button, mode in im.items():
            if button not in self._modes:
                self.set_hand_mode(button, mode)

    def _create_hand_model(self):
        # Create hand model
        name = '%s hand' % self.left_or_right
        c = self._camera
        self._hand_model = hm = HandModel(c._session, name,
                                          length=self._length, radius=self._radius,
                                          color = self._cone_color(),
                                          controller_type = self._controller_type)
        parent = c._vr_control_model_group()
        parent.add([hm])

        # Set icons for buttons
        for button, mode in self._modes.items():
            hm._set_button_icon(button, mode.icon_path)
    
    def check_if_model_closed(self):
        hm = self._hand_model
        if hm is None or hm.deleted:
            self._device_index = None
            self._hand_model = None
    
    def _cone_color(self):
        cc = self._controller_colors
        side = self.left_or_right
        color = cc[side] if side in cc else cc['default']
        from numpy import array, uint8
        rgba8 = array(color, uint8)
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
        return self._side
    
    def close(self):
        self._device_index = None
        hm = self._hand_model
        if hm:
            if not hm.deleted:
                hm.session.models.close([hm])
            self._hand_model = None

    def show_in_scene(self, show):
        self._hand_model.display = show
        
    def _update_position(self):
        '''Move hand controller model to new position.
        Keep size constant in physical room units.'''
        di = self._device_index
        hm = self._hand_model
        if di is None or hm is None:
            return
        hm.room_position = self._camera.device_position(di)
        self.update_scene_position()

    def update_scene_position(self):
        hm = self._hand_model
        if hm:
            hm.position = self._camera.room_to_scene * hm.room_position
            
    def process_event(self, e):
        if e.trackedDeviceIndex != self._device_index:
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
            ui = self._camera.user_interface
            if ui.process_hand_controller_button_event(self, b, pressed, released):
                return
        
        # Call HandMode press() or release() callback.
        if m:
            if pressed:
                self._drag_start(m, b)
            else:
                self._drag_end(m)

    def _drag_start(self, mode, button):
        mode.pressed(self._camera, self)
        mode._button_down = button
        self._active_drag_modes.add(mode)

    def _drag_end(self, mode):
        mode.released(self._camera, self)
        self._active_drag_modes.discard(mode)
        if not isinstance(mode, (ShowUIMode, MoveSceneMode, ZoomMode)):
            self._camera.user_interface.redraw_ui()
        
    def set_hand_mode(self, button, hand_mode):
        self._modes[button] = hand_mode
        hm = self._hand_model
        if hm:
            hm._set_button_icon(button, hand_mode.icon_path)

    def set_default_hand_mode(self, button):
        hand_mode = self._initial_button_modes().get(button) if self.on else None
        if hand_mode:
            self.set_hand_mode(button, hand_mode)
        elif button in self._modes:
            del self._modes[button]

    def _get_button_assignments(self):
        return tuple((button, hand_mode.name) for button, hand_mode in self._modes.items())
    def _set_button_assignments(self, button_assignments):
        ui = self._camera.user_interface
        for button, hand_mode_name in button_assignments:
            hm = ui._hand_mode_from_name(hand_mode_name)
            if hm:
                self.set_hand_mode(button, hm)
                
    button_assignments = property(_get_button_assignments, _set_button_assignments)
    '''Used for saving button assignments in sessions.'''
    
    def _thumbstick_mode(self):
        if self._is_oculus:
            import openvr
            mode = self._modes.get(openvr.k_EButton_SteamVR_Touchpad)
        else:
            mode = None
        return mode

    def thumbstick_step(self, x, y):
        if not self._thumbstick_released:
            release = self._thumbstick_release_level
            if abs(x) < release and abs(y) < release:
                self._thumbstick_released = True
                self._thumbstick_repeating = False
            elif self._thumbstick_time is not None:
                repeat = self._thumbstick_repeat_interval if self._thumbstick_repeating else self._thumbstick_repeat_delay
                from time import time
                if time() - self._thumbstick_time > repeat:
                    self._thumbstick_released = True
                    self._thumbstick_repeating = True
        if not self._thumbstick_released:
            return 0
        click = self._thumbstick_click_level
        if abs(x) < click and abs(y) < click:
            return
        self._thumbstick_released = False
        from time import time
        self._thumbstick_time = time()
        v = x if abs(x) > abs(y) else y
        step = 1 if v > 0 else -1
        return step
    
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
        
    def process_motion(self):
        if not self.on:
            return
        
        # Move hand controller model
        previous_pose = self.room_position
        self._update_position()

        # Generate mouse move event on ui panel.
        ui = self._camera.user_interface
        if ui.process_hand_controller_motion(self):
            return	# UI drag in progress.

        # Do hand controller drag when buttons pressed
        if previous_pose is not None:
            self._check_for_missing_button_release()
            pose = self.room_position
            for m in self._active_drag_modes:
                m.drag(self._camera, self, previous_pose, pose)

        # Check for Oculus thumbstick position
        ts_mode = self._thumbstick_mode()
        if ts_mode and ts_mode.uses_thumbstick():
            success, cstate = self._vr_system.getControllerState(self._device_index)
            if success:
                # On Oculus Rift S, axis 0=thumbstick, 1=trigger, 2=grip
                astate = cstate.rAxis[0]
                ts_mode.thumbstick(self._camera, self, astate.x, astate.y)

    def _check_for_missing_button_release(self):
        '''Cancel drag modes if button has been released even if we didn't get a button up event.'''
        adm = self._active_drag_modes
        if len(adm) == 0:
            return
        success, cstate = self._vr_system.getControllerState(self._device_index)
        if success:
            pressed_mask = cstate.ulButtonPressed
            for m in tuple(adm):
                # bm = openvr.ButtonMaskFromId(m._button_down)  # Routine is missing from pyopenvr
                bm = 1 << m._button_down
                if not pressed_mask & bm:
                    self._drag_end(m)

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

class HandButtons:
    def __init__(self, controller_type = 'htc vive'):
        # Cone buttons
        import openvr
        buttons = [
        ]
        if controller_type.startswith('oculus'):
            right_hand = controller_type.endswith('right')
            side, thumb_side, menu_side, stick_side = (180,110,140,80) if right_hand else (0,70,40,100)
            buttons = [
                ButtonGeometry(openvr.k_EButton_SteamVR_Trigger, z=.4, radius=.01, azimuth=270, tex_range=(.167,.333)),
                ButtonGeometry(openvr.k_EButton_SteamVR_Touchpad, z=.35, radius=.008, azimuth=stick_side, tex_range=(.333,.5)),
                ButtonGeometry(openvr.k_EButton_A, z=.47, radius=.006, azimuth=thumb_side, tex_range=(.5,.667)),
                ButtonGeometry(openvr.k_EButton_Grip, z=.6, radius=.01, azimuth=side, tex_range=(.667,.833)),
                ButtonGeometry(openvr.k_EButton_ApplicationMenu, z=.4, radius=.006, azimuth=menu_side, tex_range=(.833,1)),
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
    def __init__(self, button, z, radius, azimuth, tex_range, rise = 0.002,
                 circle_divisions = 30):
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
        self._circle_divisions = circle_divisions
        self.num_vertices = 3*circle_divisions

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
        co, so = cos(aoffset), sin(aoffset)
        from numpy import empty, float32, int32, array
        bz = array((cca*co, cca*so, -sca), float32)	# Button push axis
        bx,by = array((-so, co, 0)), array((sca*co, sca*so, cca))  # Button plane axes
        n = self._circle_divisions
        nv = self.num_vertices
        va = empty((nv,3), float32)
        na = empty((nv,3), float32)
        tc = empty((nv,2), float32)
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
            na[n+i,:] = ca*bx + sa*by
            tc[i,:] = (u0+(u1-u0)*0.5*(1+ca), v0+(v1-v0)*0.5*(1+sa))

        n2 = 2*n
        va[n:n2] = va[n2:] = va[:n]
        rise = self.rise*bz
        va[n2:] -= rise
        na[n2:] = na[n:n2]
        tc[n:] = (u0,v0)	# Sides

        vl = va.copy()
        vl += 0.1*rise
        self.vertices_lowered = vl
        vr = va.copy()
        vr += rise
        self.vertices_raised = vr

        nt = (n-2) + 2*n
        ta = empty((nt,3), int32)
        # Top of button
        for i in range(n//2-1):
            ta[2*i,:] = (i, i+1, n-1-i)
            ta[2*i+1,:] = (i+1, n-2-i, n-1-i)
        # Sides of button
        tas = ta[n-2:]
        for i in range(n):
            i1 = (i+1)%n
            tas[2*i,:] = (n+i, n2+i, n2+i1)
            tas[2*i+1,:] = (n+i, n2+i1, n+i1)

        return vr, na, tc, ta

    def set_icon_image(self, tex_rgba, icon_path, image_size):
        if icon_path:
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
        else:
            from numpy import empty, uint8
            rgba = empty((image_size, image_size, 4), uint8)
            rgba[:] = 255

        tsize = tex_rgba.shape[0]
        inset = (tsize - rgba.shape[0]) // 2
        i0 = inset
        i1 = i0 + rgba.shape[0]
        j0 = int(self.tex_range[0] * tex_rgba.shape[1]) + inset
        j1 = j0+rgba.shape[1]
        tex_rgba[i0:i1,j0:j1,:] = rgba
    
class HandMode:
    name = 'unnamed'
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
    def uses_thumbstick(self):
        return False
    def thumbstick(self, x, y):
        pass

class NoneMode(HandMode):
    name = 'none'
    @property
    def icon_path(self):
        return NoneMode.icon_location()
    @staticmethod
    def icon_location():
        from os.path import join, dirname
        return join(dirname(__file__), 'no_action.png')
    
class MoveUIMode(HandMode):
    name = 'move ui'
    def __init__(self):
        self._last_hand_position = {}	# HandController -> Place
        self._panel = None		# Move all panels if None.
        HandMode.__init__(self)
    def set_panel(self, panel):
        self._panel = panel
    def pressed(self, camera, hand_controller):
        self._last_hand_position[hand_controller] = hand_controller.room_position
    def released(self, camera, hand_controller):
        self._last_hand_position[hand_controller] = None
        self._panel = None
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
                room_motion = hrp * lhrp.inverse()
                panel = self._panel
                if panel is None:
                    ui.move(room_motion)
                else:
                    panel.move(room_motion)
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
    names = ('move scene', 'rotate', 'translate')
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

    def uses_thumbstick(self):
        return True

    def thumbstick(self, camera, hand_controller, x, y):
        from math import sqrt
        r = sqrt(x*x + y*y)
        limit = .2
        if r < limit:
            return

        f = (r-limit)/(1-limit)
        angle = f*f	# degrees
        center = _choose_zoom_center(camera)
        (vx,vy,vz) = center - camera.room_position.origin()
        from chimerax.core.geometry import normalize_vector, rotation
        horz_dir = normalize_vector((vz,0,-vx))
        from numpy import array, float32
        vert_dir = array((0,1,0),float32)  # y-axis is up in room coordinates
        axis = normalize_vector(x*vert_dir + y*horz_dir)
        move = rotation(axis, angle, center)
        camera.move_scene(move)

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
    def uses_thumbstick(self):
        return True
    def thumbstick(self, camera, hand_controller, x, y):
        limit = .2
        if abs(y) > limit:
            center = _choose_zoom_center(camera, hand_controller.tip_room_position)
            v = (y-limit if y > 0 else y+limit)/(1-limit)
            scale_factor = 1.0 - 0.02 * v * abs(v)
            _pinch_zoom(camera, scale_factor, center)

def _choose_zoom_center(camera, center = None):
    # Zoom in about center of scene if requested center point is outside scene bounding box.
    # This avoids pushing a distant scene away.
    b = camera.vr_view.drawing_bounds()
    if b and (center is None or not b.contains_point(camera.room_to_scene * center)):
        return camera.room_to_scene.inverse() * b.center()
    if center is None:
        return camera.room_position.origin()
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
    def __init__(self, mouse_mode, click_range = 5.0):
        self._mouse_mode = mouse_mode
        mouse_mode.enable()
        self.name = mouse_mode.name
        self._last_drag_room_position = None # Hand controller position at last vr_motion call
        self._laser_range = click_range	# Range for mouse mode laser clicks in scene units (Angstroms)as
        self._thumbstick_repeat_delay = 0.3	# seconds
        self._thumbstick_repeat_interval = 0.1	# seconds

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
            xyz1, xyz2 = self._picking_segment(hand_controller)
            m.vr_press(xyz1, xyz2)
        if hasattr(m, 'vr_motion'):
            self._last_drag_room_position = hand_controller.room_position if pressed else None
        if hasattr(m, 'vr_release') and not pressed:
            m.vr_release()

    def _picking_segment(self, hand_controller):
        p = hand_controller.position
        xyz1 = p * (0,0,0)
        range_scene = self._laser_range
        xyz2 = p * (0,0,-range_scene)
        return xyz1, xyz2

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

    def uses_thumbstick(self):
        return hasattr(self._mouse_mode, 'vr_thumbstick')
    
    def thumbstick(self, camera, hand_controller, x, y):
        '''Generate a mouse mode wheel event when thumbstick pushed.'''
        step = hand_controller.thumbstick_step(x, y)
        if step:
            xyz1, xyz2 = self._picking_segment(hand_controller)
            m = self._mouse_mode
            m.vr_thumbstick(xyz1, xyz2, step)

vr_hand_modes = (ShowUIMode, MoveSceneMode, ZoomMode, RecenterMode, NoneMode)

def hand_mode_names(session):
    names = set()
    for m in session.ui.mouse_modes.modes:
        names.add(m.name)
    for hm in vr_hand_modes:
        names.add(hm.name)
    return tuple(names)

def hand_mode_by_name(name):
    for mode in vr_hand_modes:
        if name == mode.name or (hasattr(mode, 'names') and name in mode.names):
            return mode
    return None

def hand_mode_icon_path(session, mode_name):
    mode = hand_mode_by_name(mode_name)
    if mode:
        path = mode.icon_location()
    else:
        mm = session.ui.mouse_modes.named_mode(mode_name)
        path = mm.icon_path if mm else None
    return path
    
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
    
