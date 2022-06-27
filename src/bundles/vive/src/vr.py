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
def vr(session, enable = None, room_position = None, mirror = None,
       gui = None, center = None, click_range = None,
       near_clip_distance = None, far_clip_distance = None,
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
    near_clip_distance : float
      Parts of the scene closer than this distance (meters) to the eye are not shown.
      Default 0.10.
    far_clip_distance : float
      Parts of the scene farther than this distance (meters) from the eye are not shown.
      Default 500.
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

    c = vr_camera(session, create = False)
    start = (session.main_view.camera is not c)

    if enable is not None:
        if enable:
            start_vr(session, multishadow_allowed, simplify_graphics)
        else:
            stop_vr(session, simplify_graphics)

    c = vr_camera(session, create = False)
    
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

    if near_clip_distance is not None:
        c.near_clip_distance = near_clip_distance

    if far_clip_distance is not None:
        c.far_clip_distance = far_clip_distance

# -----------------------------------------------------------------------------
# Assign VR hand controller buttons
#
def vr_button(session, button, mode = None, hand = None, command = None):
    '''
    Assign VR hand controller buttons

    Parameters
    ----------
    button : 'trigger', 'grip', 'touchpad', 'thumbstick', 'menu', 'A', 'B', 'X', 'Y', 'all'
      Name of button to assign.  Buttons A/B are for Oculus controllers and imply hand = 'right',
      and X/Y imply hand = 'left'
    mode : HandMode instance or 'default'
      VR hand mode to assign to button.  If mode is None then report the current mode.
    hand : 'left', 'right', None
      Which hand controller to assign.  If None then assign button on both hand controllers.
      If button is A, B, X, or Y then hand is ignored since A/B implies right and X/Y implies left.
    command : string
      Assign the button press to execute this command string.  Mode should be None or an error
      is raised.
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
    button_names = { grip: 'grip', menu: 'menu', trigger: 'trigger', touchpad: 'thumbstick', a: 'a' }

    openvr_button_ids = {
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
    openvr_buttons = openvr_button_ids[button]

    if command is not None:
        if mode is not None:
            from chimerax.core.errors import UserError
            raise UserError('vr button: Cannot specify both mode and command.')
        mode = RunCommandMode(session, command)
        
    report_modes = (mode is None and command is None)
    
    mode_names = []
    for hc in hclist:
        for button_id in openvr_buttons:
            if report_modes:
                if hc.on:
                    bname = button_names[button_id] if button == 'all' else button
                    bmode = hc.current_hand_mode(button_id)
                    mname = (bmode.name if bmode else 'none')
                    mode_names.append('%s %s = %s' % (hc.left_or_right, bname, mname))
            elif mode == 'default':
                hc.set_default_hand_mode(button_id)
            else:
                hc.set_hand_mode(button_id, mode)

    if report_modes:
        modes = ('\n' + '\n'.join(mode_names)) if len(mode_names) > 1 else ', '.join(mode_names)
        msg = 'Current VR button modes: ' + modes
        session.logger.info(msg)
        
# -----------------------------------------------------------------------------
#
def vr_room_camera(session, enable = True, field_of_view = None, width = None,
                   background_color = None, show_hands = None, show_panels = None,
                   save_position = None, tracker = None, save_tracker_mount = None):
    '''
    Mirror using fixed camera in room separate from VR headset view.

    By default VR mirroring shows the right eye view seen in the VR headset.
    This command allows instead using a camera view fixed in room coordinates.

    Parameters
    ----------
    enable : bool
      Whether to use a separate room camera for VR mirroring.
    field_of_view : float
      Horizontal field of view of room camera.  Degrees.  Default 90.
    width : float
      Width of room camera screen shown in VR in meters.  Default 1.
    background_color : Color
      Color of background in room camera rendering.  Default is dark gray.
    show_hands : bool or "toggle"
      Whether to show hand cones in room camera view.
    show_panels : bool or "toggle"
      Whether to show gui panels in room camera view.
    save_position : bool
      If true save the current camera room position for future sessions.
    tracker : bool
      Whether to set the camera position from a Vive tracker device.  Default False.
    save_tracker_mount : bool
      If true save the current relative camera position in tracker coordinates
      for future sessions.
    '''

    c = vr_camera(session)
    rc = c.enable_room_camera(enable)

    if not enable:
        loc = locals()
        for option in ('field_of_view', 'width', 'background_color',
                       'tracker', 'save_position', 'save_tracker_mount'):
            if loc.get(option) is not None:
                from chimerax.core.errors import UserError
                raise UserError('Cannot use room camera option "%s" with camera off' % option)
        return
    
    if field_of_view is not None:
        rc._field_of_view = field_of_view
    if width is not None:
        rc._camera_model.set_size(width)
    if background_color is not None:
        rc._background_color = background_color.rgba
    if show_hands is not None:
        rc.show_hands = (not rc.show_hands) if show_hands == 'toggle' else show_hands
    if show_panels is not None:
        rc.show_gui_panels = (not rc.show_gui_panels) if show_panels == 'toggle' else show_panels
    if tracker is not None:
        rc.use_tracker(tracker)
    if save_position:
        rc.save_settings(camera_position = True)
    if save_tracker_mount:
        rc.save_settings(tracker_transform = True)
        
# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, StringArg, ColorArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('mirror', BoolArg),
                              ('gui', StringArg),
                              ('center', BoolArg),
                              ('click_range', FloatArg),
                              ('near_clip_distance', FloatArg),
                              ('far_clip_distance', FloatArg),
                              ('multishadow_allowed', BoolArg),
                              ('simplify_graphics', BoolArg),
                   ],
                   synopsis = 'Start SteamVR virtual reality rendering',
                   url = 'help:user/commands/device.html#vr')
    register('vr', desc, vr, logger=logger)
    create_alias('device vr', 'vr $*', logger=logger,
                 url='help:user/commands/device.html#vr')

    button_name = EnumOf(('trigger', 'grip', 'touchpad', 'thumbstick', 'menu', 'A', 'B', 'X', 'Y', 'all'))
    desc = CmdDesc(required = [('button', button_name)],
                   optional = [('mode', VRModeArg(logger.session))],
                   keyword = [('hand', EnumOf(('left', 'right'))),
                              ('command', StringArg)],
                   synopsis = 'Assign VR hand controller buttons',
                   url = 'help:user/commands/device.html#vr-button')
    register('vr button', desc, vr_button, logger=logger)
    create_alias('device vr button', 'vr button $*', logger=logger,
                 url='help:user/commands/device.html#vr-button')

    ToggleArg = Or(EnumOf(['toggle']), BoolArg)
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('field_of_view', FloatArg),
                              ('width', FloatArg),
                              ('background_color', ColorArg),
                              ('show_hands', ToggleArg),
                              ('show_panels', ToggleArg),
                              ('save_position', BoolArg),
                              ('tracker', BoolArg),
                              ('save_tracker_mount', BoolArg)],
                   synopsis = 'Control VR room camera',
                   url = 'help:user/commands/device.html#vr-roomCamera')
    register('vr roomCamera', desc, vr_room_camera, logger=logger)
    create_alias('device vr roomCamera', 'vr roomCamera $*', logger=logger,
                 url='help:user/commands/device.html#vr-roomCamera')

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError
class VRModeArg(Annotation):
    '''Command argument for specifying VR hand controller mode.'''

    def __init__(self, session):
        Annotation.__init__(self)
        from chimerax.core.commands import quote_if_necessary
        names = list(hand_mode_names(session) + ('default',))
        names.sort()
        qnames = [quote_if_necessary(n) for n in names]
        self.name = 'one of %s' % ', '.join(qnames)
        self._html_name = 'one of %s' % ', '.join('<b>%s</b>' % n for n in qnames)
        
    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import EnumOf
        mode_arg = EnumOf(hand_mode_names(session) + ('default',))
        mode_name, used, rest = mode_arg.parse(text, session)
        if mode_name == 'default':
            hm = 'default'
        else:
            c = vr_camera(session)
            hm = c.user_interface._hand_mode_from_name(mode_name)
            if hm is None:
                raise AnnotationError('Unknown VR hand mode "%s"' % mode_name)
        return hm, used, rest

# -----------------------------------------------------------------------------
#
def start_vr(session, multishadow_allowed = False, simplify_graphics = True,
             label_reorient = 45, zone_label_color = (255,255,0,255), zone_label_background = (0,0,0,255)):

    v = session.main_view
    if not multishadow_allowed and v.lighting.multishadow > 0:
        from chimerax.core.commands import run
        run(session, 'lighting simple')

    if simplify_graphics:
        from chimerax.std_commands.graphics import graphics_quality
        graphics_quality(session, total_atom_triangles=1000000, total_bond_triangles=1000000)

    # Use only 8 shadow directions for faster rendering.
    from chimerax.std_commands.lighting import lighting_settings
    lighting_settings(session).lighting_multishadow_directions = 8
        
    # Don't continuously reorient labels.
    from chimerax.label.label3d import label_orient
    label_orient(session, label_reorient)

    # Make zone mouse mode labels easier to read
    from chimerax.zone.zone import zone_setting
    zone_setting(session, label_color = zone_label_color,
                 label_background_color = zone_label_background, save = False)
    
    c = vr_camera(session)
    if c is session.main_view.camera:
        return

    try:
        import openvr
    except Exception as e:
        from chimerax.core.errors import UserError
        raise UserError('Failed to import OpenVR module: %s' % str(e)) from e

    import sys
    if sys.platform == 'darwin':
        # SteamVR on Mac is older then what PyOpenVR expects.
        openvr.IVRSystem_Version = "IVRSystem_019"
        openvr.IVRCompositor_Version = "IVRCompositor_022"
        
    try:
        c.start_vr()
    except openvr.OpenVRError as e:
        if 'error number 108' in str(e):
            msg = ('The VR headset was not detected.\n' +
                   'Possibly a cable to the VR headset is not plugged in.\n' +
                   'If the headset is a Vive Pro, the link box may be turned off.\n' +
                   'If using a Vive Pro wireless adapter it may not be powered on.')
        elif 'InterfaceNotFound' in str(e):
            msg = ('Your installed SteamVR runtime does not support the requested version.\n' +
                   'You probably need to update SteamVR by starting the Steam application.\n')
        else:
            msg = ('Failed to initialize OpenVR.\n' +
                   'Possibly SteamVR is not installed or it failed to start.')
        from chimerax.core.errors import UserError
        raise UserError('%s\n%s' % (msg, str(e))) from e

    session.main_view.camera = c

    # VR gui cannot display a native file dialog.
    from chimerax.open_command import set_use_native_open_file_dialog
    set_use_native_open_file_dialog(False)
    
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
    
    from chimerax.graphics import MonoCamera
    v = session.main_view
    v.camera = MonoCamera()

    session.update_loop.set_redraw_interval(10)
    
    if simplify_graphics:
        from chimerax.std_commands.graphics import graphics_quality
        graphics_quality(session, total_atom_triangles=5000000, total_bond_triangles=5000000)

    # Continuously reorient labels.
    from chimerax.label.label3d import label_orient
    label_orient(session, 0)

    # Go back to 64 shadow directions
    from chimerax.std_commands.lighting import lighting_settings
    lighting_settings(session).lighting_multishadow_directions = 64
    
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
from chimerax.graphics import Camera
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
        self._tracker_device_index = None	# Vive tracker
        
        self.user_interface = UserInterface(self, session)
        self._vr_model_group = None	# Grouping model for hand controllers and UI models
        self._vr_model_group_id = 100	# Keep VR model group at bottom of model panel

        self._mirror = True		# Whether to render to desktop graphics window.
        self._room_camera = None	# RoomCamera, fixed view camera independent of VR headset

        from chimerax.geometry import Place
        self.room_position = Place()	# ChimeraX camera coordinates to room coordinates
        self._room_to_scene = None	# Maps room coordinates to scene coordinates
        self._z_near = 0.1		# Meters, near clip plane distance
        self._z_far = 500.0		# Meters, far clip plane distance
        # TODO: Scaling models to be huge causes clipping at far clip plane.

        self._vr_system = None		# openvr.IVRSystem instance
        self._new_frame_handler = None
        self._app_quit_handler = None

    def start_vr(self):
        if self._vr_system is not None:
            return	# VR is already started
        
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
        self._set_projection_matrices()

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
        self._new_frame_handler = t.add_handler('new frame', self._next_frame)

        # Assign hand controllers
        self._find_hand_controllers()
        
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

    def _set_projection_matrices(self):
        z_near, z_far = self._z_near, self._z_far
        vrs = self._vr_system
        import openvr
        pl = vrs.getProjectionMatrix(openvr.Eye_Left, z_near, z_far)
        self._projection_left = hmd44_to_opengl44(pl)
        pr = vrs.getProjectionMatrix(openvr.Eye_Right, z_near, z_far)
        self._projection_right = hmd44_to_opengl44(pr)

    def _get_near_clip_distance(self):
        return self._z_near
    def _set_near_clip_distance(self, near):
        self._z_near = near
        self._set_projection_matrices()
    near_clip_distance = property(_get_near_clip_distance, _set_near_clip_distance)

    def _get_far_clip_distance(self):
        return self._z_far
    def _set_far_clip_distance(self, far):
        self._z_far = far
        self._set_projection_matrices()
    far_clip_distance = property(_get_far_clip_distance, _set_far_clip_distance)
        
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

    @property
    def room_camera(self):
        return self._room_camera

    @property
    def have_room_camera(self):
        return self._room_camera is not None

    @property
    def have_tracker(self):
        return (self._tracker_device_index is not None
                or self._find_tracker() is not None)
    
    def tracker_room_position(self):
        i = self._tracker_device_index
        if i is None:
            i = self._find_tracker()
            if i is None:
                return None
        return self.device_position(i)

    def _find_tracker(self):
        import openvr
        for device_id in range(openvr.k_unMaxTrackedDeviceCount):
            if self._device_type(device_id) == 'tracker' and self._device_connected(device_id):
                self._tracker_device_index = device_id
                return device_id
        return None
        
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
            if g is None or g.deleted:
                b = self.vr_view.drawing_bounds()
            else:
                # Need to exclude UI from bounds.
                top_models = self._session.models.scene_root_model.child_models()
                from chimerax.geometry import union_bounds
                b = union_bounds(m.bounds() for m in top_models
                                 if m.display and m.id[0] != g.id[0] and
                                 not getattr(m, 'skip_bounds', False))
        if b:
            scene_size = b.width()
            scene_center = b.center()
        else:
            scene_size = 1
            from numpy import zeros, float32
            scene_center = zeros((3,), float32)
        # First apply scene shift then scene scale to get room coords
        from chimerax.geometry import translation, scale
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
        self._compositor = None
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
        p = self._poses[device_index]
        if not p.bPoseIsValid:
            return None
        dp = p.mDeviceToAbsoluteTracking
        return hmd34_to_position(dp)
    
    def _next_frame(self, *_):
        if not self.active:
            # If the session camera is changed from the VR camera
            # without calling the VR camera close method (should
            # never happen) then close the VR camera.  Othewise all
            # the VR updating will continue even when the camera is
            # not in use.
            self.close()
            return
        
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
        from chimerax.geometry import scale
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
            type = e.eventType
            if type == openvr.VREvent_TrackedDeviceActivated:
                i = e.trackedDeviceIndex
                dtype = self._device_type(i)
                if dtype == 'controller':
                    self._hand_controller_enabled(i)
                elif dtype == 'tracker':
                    self._tracker_device_index = i
            elif type == openvr.VREvent_TrackedDeviceDeactivated:
                i = e.trackedDeviceIndex
                dtype = self._device_type(i)
                if dtype == 'controller':
                    self._hand_controller_disabled(e.trackedDeviceIndex)
                elif dtype == 'tracker':
                    self._tracker_device_index = None
            else:
                for hc in self.hand_controllers():
                    hc.process_event(e)

    def _device_type(self, device_index):
        vrs = self._vr_system
        c = vrs.getTrackedDeviceClass(device_index)
        import openvr
        tmap = {openvr.TrackedDeviceClass_Controller: 'controller',
                openvr.TrackedDeviceClass_GenericTracker: 'tracker',
                openvr.TrackedDeviceClass_HMD: 'hmd'}
        return tmap.get(c, 'unknown')

    def _device_connected(self, device_index):
        vrs = self._vr_system
        return vrs.isTrackedDeviceConnected(device_index)

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
            ss = self.scene_scale
            if ss == 0:
                v = camera_position
            else:
                t = es.scale_translation(1/ss)
                v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera.'''
        draw_desktop = (self._room_camera and self._session.ui.main_window.graphics_window.is_drawable)
        return 3 if draw_desktop else 2

    def view_width(self, point):
        fov = 100	# Effective field of view, degrees
        from chimerax.graphics.camera import perspective_view_width
        return perspective_view_width(point, self.position.origin(), fov)

    def view_all(self, bounds, window_size = None, pad = 0):
        fov = 100	# Effective field of view, degrees
        from chimerax.graphics.camera import perspective_view_all
        p = perspective_view_all(bounds, self.position, fov, window_size, pad)
        self._move_camera_in_room(p)
        self.fit_scene_to_room(bounds)

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        if view_num == 2:
            # Use near_far_clip in meters in room rather than actual scene bounds
            # because scene bounds don't include hand controllers, vr user interface panels
            # and multi-person head models so those get clipped in the room camera view
            # if the data model bounds are too small.
            ss = self.scene_scale
            nf = (self._z_near/ss, self._z_far/ss) if ss > 0 else near_far_clip
            p = self._room_camera.projection_matrix(nf, view_num, window_size)
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
            from chimerax.graphics.drawing import draw_overlays
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
            from chimerax.graphics import Texture, opengl
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
            from chimerax.graphics.drawing import _texture_drawing
            self._texture_drawing = td = _texture_drawing(texture)
            td.opaque_texture = True
        else:
            td.texture = texture
        window_size = self.render.render_size()
        from chimerax.graphics.drawing import match_aspect_ratio
        match_aspect_ratio(td, window_size)
        return td

    def do_swap_buffers(self):
        return self.mirror

    def hand_controllers(self):
        return self._hand_controllers

    def _hand_controller_enabled(self, device_id):
        self._assign_hand_controller(device_id)

    def _hand_controller_disabled(self, device_id):
        for hc in self._hand_controllers:
            if hc.device_index == device_id:
                hc.device_index = None
        
    def _find_hand_controllers(self):
        # Find hand controllers that are turned on.
        import openvr
        for device_id in range(openvr.k_unMaxTrackedDeviceCount):
            self._assign_hand_controller(device_id)

    def _assign_hand_controller(self, device_id):
        d = device_id
        if self._device_type(d) != 'controller':
            return
        vrs = self._vr_system
        if not vrs.isTrackedDeviceConnected(d):
            return
        left_or_right = self._controller_left_or_right(d)
        assigned = False
        for hc in self._hand_controllers:
            if hc.left_or_right == left_or_right:
                if hc.device_index is None:
                    hc.device_index = d
                    assigned = True
        if not assigned:
            # This happens when a second right or left controller activates.
            # In this case don't believe its purported right/left.
            for hc in self._hand_controllers:
                if hc.device_index is None:
                    hc.device_index = d
                    assigned = True

    def _controller_left_or_right(self, device_index):
        vrs = self._vr_system

        import openvr
        left_id = vrs.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_LeftHand)
        if device_index == left_id:
            return 'left'
        right_id = vrs.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_RightHand)
        if device_index == right_id:
            return 'right'

        # Above left and right role are 2**32-1 for Oculus when first started.
        # Try looking at the controller name.
        model_name = vrs.getStringTrackedDeviceProperty(device_index,
                                                        openvr.Prop_RenderModelName_String)
        if model_name.endswith('right'):
            return 'right'
        if model_name.endswith('left'):
            return 'left'

        # Don't know whether left or right.
        return 'right'

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
            if hc != controller and hc.on:
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
        self._session = parent.session
        self._framebuffer = None	# Framebuffer for rendering room camera view.
        self._camera_model = None
        self._field_of_view = 90	# Degrees.  Horizontal.
        self._background_color = (.1,.1,.1,1)	# RGBA, float 0-1
        self._settings = None		# Saved preferences, room position.
        self.is_rendering = False	# Allows hand, gui, and screen models to hide themselves
        self.show_hands = True		# Whether to show hand cones in room camera view
        self.show_gui_panels = True	# Whether to show VR gui panel in room camera view

        # Depiction of camera in VR scene.
        render.make_current()	# Texture is allocated when framebuffer created.
        texture = self.framebuffer(render).color_texture
        self._camera_model = self._create_camera_model(parent, room_to_scene, texture)

    def delete(self, render):
        self._delete_framebuffer(render)

    def close(self, render):
        self._delete_framebuffer(render)
        cm = self._camera_model
        if cm:
            if not cm.deleted:
                cm.delete()
            self._camera_model = None
        
    @property
    def enabled(self):
        return self._camera_model is not None

    @property
    def camera_position(self):
        cm = self._camera_model
        if cm is None or cm.deleted:
            from chimerax.geometry import Place
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
        from chimerax.graphics.camera import perspective_projection_matrix
        return perspective_projection_matrix(fov, window_size, near_far_clip, pixel_shift)
    
    def _create_camera_model(self, parent, room_to_scene, texture):
        cm = RoomCameraModel('Room camera', parent.session, texture, room_to_scene)
        cm.room_position = self._initial_room_position()
        parent.add([cm])
        return cm
    
    def _initial_room_position(self):
        s = self._saved_settings()
        from chimerax.geometry import Place
        p = Place(s.independent_camera_position)
        return p

    def use_tracker(self, use):
        ses = self._session
        cm = self._camera_model
        if cm is None:
            ses.logger.warning('Room camera does not exist.  Cannot enable tracker.')
            return
        cam = ses.main_view.camera
        if not hasattr(cam, 'tracker_room_position'):
            ses.logger.warning('Cannot use room camera tracker before VR camera enabled')
            return
        p = cam.tracker_room_position()
        if p is None:
            ses.logger.warning('No Vive tracker found.')
            return
        tt = self._tracker_transform()
        cm.room_position = p*tt

    def _tracker_transform(self):
        s = self._saved_settings()
        from chimerax.geometry import Place
        p = Place(s.tracker_transform)
        return p

    def _saved_settings(self):
        if self._settings is None:
            from chimerax.geometry import translation
            # Centered 1.5 meters off floor, 2 meters from center
            default_position = translation((0, 1.5, 2))
            m = tuple(tuple(row) for row in default_position.matrix)
            from chimerax.core.settings import Settings
            class _VRRoomCameraSettings(Settings):
                EXPLICIT_SAVE = {
                    'independent_camera_position': m,
                    'tracker_transform': ((1,0,0,0),(0,1,0,0),(0,0,1,0)),
                }
            self._settings = _VRRoomCameraSettings(self._session, "vr_room_camera")
        return self._settings

    def save_settings(self, camera_position = False, tracker_transform = False):
        cm = self._camera_model
        if cm is None:
            return
        settings = self._saved_settings()
        if camera_position:
            m = tuple(tuple(row) for row in cm.room_position.matrix)
            settings.independent_camera_position = m
        if tracker_transform:
            cam = self._session.main_view.camera
            if hasattr(cam, 'tracker_room_position'):
                trp = cam.tracker_room_position()
                if trp:
                    tt = trp.inverse() * cm.room_position
                    tm = tuple(tuple(row) for row in tt.matrix)
                    settings.tracker_transform = tm
        settings.save()
        
    def start_rendering(self, render):
        fb = self.framebuffer(render)
        render.push_framebuffer(fb)

        # Set paramters for mixed reality blending.
        render.mix_video = True  # For making mixed reality videos

        # Allow hand, gui, and screen models to detect if room
        # camera is rendering so they can hide themselves.
        self.is_rendering = True
        
        # Make background contrast with room background so vr user can see boundary.
        render.set_background_color(self._background_color)

    def finish_rendering(self, render):
        # Turn off mixed reality blending.
        render.mix_video = False

        # Reenable hand, gui and screen rendering for non-room-camera views.
        self.is_rendering = False

    def framebuffer(self, render):
        rfb = render.default_framebuffer()
        tw,th = rfb.width, rfb.height
        fb = self._framebuffer
        if fb is None or fb.width != tw or fb.height != th:
            self._delete_framebuffer(render)
            from chimerax.graphics import Texture, opengl
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
    SESSION_SAVE = False

    def __init__(self, name, session, texture, room_to_scene, width = 1):
        '''Width in meters.'''
        self._last_room_to_scene = room_to_scene
        self._width = width

        Model.__init__(self, name, session)

        self.casts_shadows = False
        self.skip_bounds = True   # "view all" command excludes room camera

        # Avoid camera disappearing when far from models
        self.allow_depth_cue = False

        # Avoid clip planes hiding the camera screen.
        self.allow_clipping = False

        self.color = (255,255,255,255)	# Don't modulate texture colors.
        self.use_lighting = False
        self.texture = texture
        self.opaque_texture = True
        self.set_size(width)

    def delete(self):
        cam = self.session.main_view.camera
        Model.delete(self)
        if isinstance(cam, SteamVRCamera):
            cam.enable_room_camera(False)
            
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
        if self._hide_camera():
            return
        # TODO: Graphics is drawn in opaque draw pass because self.opaque_texture is True
        # but the texture may have transparent alpha values.  The drawing code uses alpha
        # blending even in the opaque pass so we need to turn it off here.  Maybe draw pass
        # code should be disabling alpha blending.
        renderer.enable_blending(False)
        Model.draw(self, renderer, draw_pass)
        renderer.enable_blending(True)

    def _hide_camera(self):
        '''
        Returns true if the room camera is currently being rendered.
        This is used to suppress screen drawing so it does not block the view.
        '''
        c = self.session.main_view.camera
        if isinstance(c, SteamVRCamera):
            rc = c.room_camera
            return rc and rc.is_rendering
        return False
    
    def update_scene_position(self, new_rts):
        old_rts = self._last_room_to_scene
        self._last_room_to_scene = new_rts
        move = new_rts * old_rts.inverse()
        mpos = move * self.position
        # Need to remove scale factor.
        from chimerax.geometry import norm, Place
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
        self._resizing_panel = None	# (Panel, HandController) when panel being resized by click and drag on titlebar
        self._tool_show_handler = None
        self._tool_hide_handler = None

        # Buttons that can be pressed on user interface.
        import openvr
        self.buttons = (openvr.k_EButton_SteamVR_Trigger, openvr.k_EButton_Grip, openvr.k_EButton_SteamVR_Touchpad,
                        openvr.k_EButton_A)
        
    def close(self):
        ui = self._ui_model
        if ui:
            self._session.models.close([ui])
            self._ui_model = None

        for h in (self._tool_show_handler, self._tool_hide_handler):
            if h is not None:
                triggers = self._session.ui.triggers
                triggers.remove_handler(h)
        self._tool_show_handler = self._tool_hide_handler = None

        self._panels = []

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
        tool_names = self._gui_tool_names
        if tool_names:
            tool_wins = []
            for tool_name in tool_names:
                twins = _tool_windows(tool_name, self._session)
                if len(twins) == 0:
                    self._session.logger.warning('VR user interface tool "%s" not running'
                                                 % tool_name)
                tool_wins.extend(twins)
        else:
            # Use all shown tools.
            exclude_tools = set(['Command Line Interface'])
            tool_wins = [tw for tw in _tool_windows(None, self._session)
                         if tw.tool_instance.tool_name not in exclude_tools]
        tool_wins.sort(key = _tool_y_position)
        tpanels = [Panel(tw, ui, self, add_titlebar = (tw.tool_instance.tool_name != 'Toolbar'))
                   for tw in tool_wins if tw.shown]
        panels.extend(tpanels)

        # Position panels on top of each other
        self._stack_panels(panels)

        # Monitor when windows are shown and hidden.
        triggers = self._session.ui.triggers
        self._tool_show_handler = triggers.add_handler('tool window show',
                                                       self._tool_window_show)
        self._tool_hide_handler = triggers.add_handler('tool window hide',
                                                       self._tool_window_hide)
        return panels

    def _stack_panels(self, panels):
        sep = self._panel_y_spacing
        dz = self._panel_z_spacing
        spanels = [p for p in panels if not p.is_menu()]
        h = sum(p.size[1] for p in spanels) + (len(spanels)-1)*sep
        # Stack panels.
        y = h/2
        z = -dz
        for p in spanels:
            h = p.size[1]
            y -= 0.5*h
            p.center = (0,y,z)
            y -= 0.5*h + sep
            z -= dz

        if len(panels) > len(spanels):
            # Position menu panels.
            mpanels = [p for p in panels if p.is_menu()]
            for mp in mpanels:
                mp.position_menu_over_parent(spanels)
                
    def _tool_window_show(self, trig_name, tool_window):
        self._add_tool_panel(tool_window)

    def _tool_window_hide(self, trig_name, tool_window):
        self._delete_tool_panel(tool_window)

    def _add_tool_panel(self, tool_window):
        if not self._find_tool_panel(tool_window):
            is_toolbar = (tool_window.tool_instance.tool_name == 'Toolbar')
            p = Panel(tool_window, self._ui_model, self, add_titlebar = not is_toolbar)
            self._add_panels([p])
            self.redraw_ui()

    def _add_panels(self, panels):
        z = max([p.center[2]+p.thickness for p in self._panels],
                default = 0)
        for p in panels:
            if not p.is_menu():
                z += p.thickness
                p.center = (0,0,z)
                z += p.thickness
        self._panels.extend(panels)

    def _user_moved_panels(self):
        for p in self._panels:
            if p.position.rotation_angle() != 0:
                return True
        return False
            
    def _check_for_new_panels(self):
        # Add new panels for newly appeared top level widgets.
        from Qt.QtWidgets import QDockWidget, QMainWindow, QMenu
        tw = [w for w in self._session.ui.topLevelWidgets()
              if w.isVisible() and not isinstance(w, (QDockWidget, QMainWindow))]
        wset = set(p._widget for p in self._panels)
        neww = [w for w in tw if w not in wset]
        newp = [Panel(w, self._ui_model, self, tool_name = w.windowTitle(),
                      add_titlebar = not isinstance(w, QMenu))
                for w in neww]
        self._add_panels(newp)
        
        for p in newp:
            if p.is_menu():
                p.position_menu_over_parent(self._panels)

        # Show rapid access panel
        w = self._session.ui.main_window.rapid_access
        if w.isVisible() and w not in wset:
            p = Panel(w, self._ui_model, self, tool_name = 'Recent Files',
                      add_titlebar = True)
            self._add_panels([p])
            
        if neww:
            self.redraw_ui()

        # Remove closed panels
        for p in tuple(self._panels):
            try:
                vis = p._widget.isVisible()
            except Exception:
                vis = False	# Panel destroyed
            if not vis:
                self._delete_panel(p)

    def _find_tool_panel(self, tool_window):
        for p in self._panels:
            if tool_window is p._tool_window:
                return p
        return None

    def _delete_tool_panel(self, tool_window):
        p = self._find_tool_panel(tool_window)
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
            self._resizing_panel = None
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
                hc._dispatch_event(self._move_ui_mode, HandButtonEvent(hc, b, released = True))
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
                    panel.close_widget()
                    self.redraw_ui()
                elif panel.clicked_on_resize_button(window_xy):
                    panel.undock()
                    self._resizing_panel = (panel, hc)
                elif panel.clicked_on_title_bar(window_xy):
                    # Drag on title bar moves VR gui
                    self._move_gui.add((hc,b))
                    mum = self._move_ui_mode
                    mum.set_panel(panel)
                    hc._dispatch_event(mum, HandButtonEvent(hc, b, pressed=True))
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

        if self._resize_panel(hc):
            return True
        
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
        p = hc.tip_room_position
        if p is not None:
            self._highlight_button(p, hc)

        return False

    def _resize_panel(self, hand_controller):
        if self._resizing_panel is None:
            self._last_panel_resize_position = None
            return False

        panel, resize_hc = self._resizing_panel
        if hand_controller is not resize_hc:
            return False

        hpos = hand_controller.room_position.origin()
        window_xy, z_offset = panel._panel_click_position(hpos)
        if window_xy is None:
            return True
        
        lhpos = getattr(self, '_last_panel_resize_position', None)
        self._last_panel_resize_position = hpos
        if lhpos is None:
            return True
        
        last_xy, z_offset = panel._panel_click_position(lhpos)
        if last_xy is None:
            return True
        
        dx,dy = [x-xl for x,xl in zip(window_xy, last_xy)]
        rx,ry = -2*int(round(dx)), -2*int(round(dy))
        panel.resize_widget(rx, ry)
        
        if rx == 0 and ry == 0:
            self._last_panel_resize_position = lhpos  # Accumulate motion

        return True
            
    def _highlight_button(self, room_point, highlight_id):
        window_xy, panel = self._click_position(room_point)
        if panel:
            widget, wpos = panel.clicked_widget(window_xy)
            from Qt.QtWidgets import QAbstractButton, QTabBar
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

    def redraw_ui(self, delay = True, delay_frames = None):
        if delay:
            frames = self._update_delay if delay_frames is None else delay_frames
            self._update_later = frames
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
        for panel in tuple(self._panels):
            if panel._window_closed():
                self._delete_panel(panel)
            else:
                panel._update_image()
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
                m = MouseMode(mouse_mode)
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
        uim = UIModel(ses, self._ui_model_closed)
        ses.models.add([uim], parent = parent)
        return uim

    def _ui_model_closed(self):
        self._ui_model = None
        self.close()
        
    def display_ui(self, hand_room_position, camera_position):
        rp = hand_room_position
        # Orient horizontally and facing camera.
        view_axis = camera_position.origin() - rp.origin()
        from chimerax.geometry import orthonormal_frame, translation
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

from chimerax.core.models import Model
class UIModel(Model):
    def __init__(self, session, close_cb = None):
        self._close_cb = close_cb
        Model.__init__(self, 'User interface', session)
        self.color = (255,255,255,255)
        self.use_lighting = False
        self.casts_shadows = False
        self.pickable = False
        self.SESSION_SAVE = False
    def delete(self):
        if self._close_cb is not None:
            self._close_cb()
        Model.delete(self)
        
class Panel:
    '''The VR user interface consists of one or more rectangular panels.'''
    def __init__(self, tool_or_widget, drawing_parent, ui,
                 tool_name = None, pixel_size = 0.001, add_titlebar = False):
        from chimerax.ui.gui import ToolWindow
        if isinstance(tool_or_widget, ToolWindow) or hasattr(tool_or_widget, 'tool_instance'):
            # TODO: Remove test for tool_instance attribute
            # needed to work around bug #2875
            tw = tool_or_widget
            self._tool_window = tw
            self._widget = tw.ui_area
            if tool_name is None:
                tool_name = tw.tool_instance.tool_name
        else:
            self._tool_window = None
            self._widget = tool_or_widget	# This Qt widget is shown in the VR panel.
        self._ui = ui			# UserInterface instance
        self._tool_name = tool_name	# Name of tool instance
        th = 20 if add_titlebar else 0
        self._titlebar_height = th      # Added titlebar height in pixels
        w,h = self._panel_size
        self._size = (pixel_size*w, pixel_size*h) # Billboard width, height in room coords, meters.
        self._pixel_size = pixel_size	# In meters.

        self._last_image_rgba = None
        self._ui_click_range = 0.05 	# Maximum distance of click from plane, room coords, meters.
        self._button_rise = 0.01	# meters rise when pointer over button
        self._panel_thickness = 0.01	# meters

        # Drawing that renders this panel.
        self._panel_drawing = d = PanelDrawing()
        drawing_parent.add_drawing(d)

    @property
    def widget(self):
        w = self._widget
        try:
            w.width()
        except Exception:
            w = None	# Widget was deleted.
        return w

    def _get_center(self):
        pd = self._panel_drawing
        return pd.position.origin() if pd else None
    def _set_center(self, center):
        pd = self._panel_drawing
        if pd:
            from chimerax.geometry import translation
            pd.position = translation(center)
    center = property(_get_center, _set_center)

    @property
    def thickness(self):
        return self._panel_thickness
    
    @property
    def position(self):
        pd = self._panel_drawing
        from chimerax.geometry import Place
        return pd.position if pd else Place()
    
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

    def resize_widget(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        w = self._widget
        top = w.window()
        s = top.size()
        xs = max(1, s.width() + dx)
        ys = max(1, s.height() + dy)
        top.resize(xs, ys)
        self._update_image()

    def undock(self):
        tw = self._tool_window
        if tw is not None and not tw.floating:
            tw.floating = True

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
            from chimerax.geometry import translation
            pd.position = translation(shift) * pd.position
            
    def _panel_click_position(self, room_point):
        ui = self._panel_drawing
        if ui is None:
            return None, None
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

    def _update_image(self):
        rgba = self._panel_image()
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
            from chimerax.graphics import Texture
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

    def _panel_image(self):
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
        size = pixmap.size()
        if size.width() == 0 or size.height() == 0:
            return None
        im = pixmap.toImage()
        from chimerax.graphics.drawing import qimage_to_numpy
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
        trgba[h:,:,:] = background_color

        # Add resize button
        rs_sign = '\u21F1'	# Unicode resize symbol
        rs_rgba = self._icon_image('resize', rs_sign, title_color, th, background_color)
        rw = min(rs_rgba.shape[1], trgba.shape[1])
        trgba[h:,:rw,:] = rs_rgba[:,:rw,:]

        # Add title text
        title = self.name
        if title:
            from chimerax.graphics import text_image_rgba
            title_rgba = text_image_rgba(title, title_color, th, 'Arial',
                                         background_color = background_color,
                                         xpad = 8, ypad = 4, pixels = True)
            tw = min(title_rgba.shape[1], trgba.shape[1]-rw)
            trgba[h:,rw:rw+tw,:] = title_rgba[:,:tw,:]

        # Add close button
        x_sign = '\u00D7'	# Unicode multiply symbol
        x_rgba = self._icon_image('close', x_sign, title_color, th, background_color, xpad = 6)
        xw = min(x_rgba.shape[1], trgba.shape[1])
        trgba[h:,-xw:,:] = x_rgba[:,:xw,:]

        return trgba

    def _icon_image(self, name, character, color, height, background_color, xpad = 0):
        attr = '_%s_icon_rgba' % name
        icon_rgba = getattr(self, attr, None)
        if icon_rgba is None:
            from chimerax.graphics import text_image_rgba
            icon_rgba = text_image_rgba(character, color, height, 'Arial',
                                     background_color = background_color,
                                     xpad = xpad, pixels = True)
            setattr(self, attr, icon_rgba)
        return icon_rgba

    def is_menu(self):
        from Qt.QtWidgets import QMenu
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
        from Qt.QtCore import QPoint
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
        'Type can be "press" or "release" or "move".'
        self._move_mouse_pointer(window_xy)
        w = self._post_mouse_event(type, window_xy)
        if w:
            if type == 'press':
                self._show_pressed_button(w)
            if type == 'release':
                self._show_pressed_button(w, pressed = False)
                self._ui.redraw_ui()
            return True
        return False

    def _move_mouse_pointer(self, window_xy):
        # Sometimes Windows uses the mouse position instead of the button
        # event coordinates, for instance when handling cascaded menus.
        # Details in ChimeraX bug #2848.
        # So move the mouse pointer to the position of the virtual button event.
        pw = self.widget
        if pw is not None:
            x,y = window_xy
            from Qt.QtCore import QPoint
            wp = QPoint(int(x), int(y))
            p = pw.mapToGlobal(wp)
            from Qt.QtGui import QCursor
            QCursor.setPos(p)

    def _post_mouse_event(self, type, window_xy):
        '''Type is "press", "release" or "move".'''
        w, pos = self.clicked_widget(window_xy)
        if w is None or pos is None:
            return w
        from Qt.QtCore import Qt, QEvent
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
        from Qt.QtCore import QPoint, QPointF
        screen_pos = QPointF(w.mapToGlobal(QPoint(int(pos.x()), int(pos.y()))))
        from Qt.QtGui import QMouseEvent
        me = QMouseEvent(et, pos, screen_pos, button, buttons, Qt.NoModifier)
        self._ui._session.ui.postEvent(w, me)
        return w

    def clicked_widget(self, window_xy):
        # Input window_xy coordinates are in top level window that panel is in.
        # Returns clicked widget and (x,y) in widget pixel coordinate system.
        pw = self.widget
        if pw is None:
            return None, None
        from Qt.QtCore import QPoint, QPointF
        x,y = window_xy
        pwp = QPoint(int(x), int(y))
        w = pw.childAt(pwp)	# Works even if widget is covered.
        if w is None:
            return pw, QPointF(x,y)
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

    def clicked_on_resize_button(self, window_xy):
        th = self._titlebar_height
        if th > 0:
            x,y = window_xy
            return y < 0 and x <= th
        return False

    def clicked_on_close_button(self, window_xy):
        th = self._titlebar_height
        if th > 0:
            x,y = window_xy
            return y < 0 and x >= self._panel_size[0]-th
        return False

    def close_widget(self):
        '''Called when close button on panel titlebar pressed.'''
        tw = self._tool_window
        if tw:
            self._ui._session.ui.main_window.close_request(tw)
        else:
            self.widget.close()
        
    def clicked_on_title_bar(self, window_xy):
        th = self._titlebar_height
        if th > 0:
            return window_xy[1] < 0
        w, pos = self.clicked_widget(window_xy)
        from Qt.QtWidgets import QMenuBar, QDockWidget
        if isinstance(w, QMenuBar):
            from Qt.QtCore import QPoint
            ipos = QPoint(int(pos.x()),int(pos.y()))
            if w.actionAt(ipos) is None:
                return True
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        return isinstance(w, (QDockWidget, TabbedToolbar))
                           
    def clicked_mouse_mode(self, window_xy):
        w, pos = self.clicked_widget(window_xy)
        from Qt.QtWidgets import QToolButton
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
        
        from Qt.QtCore import QPoint
        pos = w.mapToGlobal(QPoint(0,0))
        ppos = p._widget.mapFromGlobal(pos)
        y = ppos.y() + p._titlebar_height
        ps = p._pixel_size
        pw,ph = p._size
        sw,sh = self._size
        offset = (ppos.x()*ps + sw/2 - pw/2, -(y*ps + sh/2 - ph/2), .01)
        pd = self._panel_drawing
        from chimerax.geometry import translation
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

from chimerax.graphics import Drawing
class PanelDrawing(Drawing):
    '''Draws a single gui panel as a textured rectangle.'''
    def __init__(self, name = 'VR UI panel'):
        Drawing.__init__(self, name)
        self.color = (255,255,255,255)
        self.use_lighting = False
        self.casts_shadows = False
        self.skip_bounds = True		# Panels should not effect view all command.
        self.allow_depth_cue = False	# Avoid panels fading out far from models.
        self.allow_clipping = False	# Avoid clip planes hiding panels

    def draw(self, renderer, draw_pass):
        if not self._hide_panel():
            Model.draw(self, renderer, draw_pass)

    def _hide_panel(self):
        '''
        Returns true if the room camera is currently being rendered
        and the GUI panels are to be hidden.
        '''
        c = self.parent.session.main_view.camera
        if isinstance(c, SteamVRCamera):
            rc = c.room_camera
            return rc and rc.is_rendering and not rc.show_gui_panels
        return False

def _ancestor_widgets(w):
    alist = []
    p = w
    while True:
        p = p.parentWidget()
        if p is None:
            return alist
        alist.append(p)

def _tool_y_position(tool_window):
    w = tool_window.ui_area
    from Qt.QtCore import QPoint
    return w.mapToGlobal(QPoint(0,0)).y()

def _tool_windows(name, session):
    return [ti.tool_window for ti in session.tools.list()
            if hasattr(ti, 'tool_window')]
            
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

    @property
    def on(self):
        return self._device_index is not None

    def _get_device_index(self):
        return self._device_index
    def _set_device_index(self, device_index):
        if device_index == self._device_index:
            return
        self._device_index = device_index
        if device_index is not None:
            self._set_controller_type()
            if self.hand_model is None:
                self._create_hand_model()
            self._set_initial_button_assignments()
        else:
            self._close_hand_model()
    device_index = property(_get_device_index, _set_device_index)

    @property
    def hand_model(self):
        hm = self._hand_model
        if hm and hm.deleted:
            self._hand_model = hm = None
        return hm
    
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
            right = (self.left_or_right == 'right')
            thumbstick_mode = ZoomMode() if right else MoveSceneMode()
            ax_mode = ZoomMode() if right else RecenterMode()
            initial_modes = {
                menu: ShowUIMode(),
                trigger: MoveSceneMode(),
                grip: MoveSceneMode(),
                a: ax_mode,
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

        return hm
    
    def _cone_color(self):
        cc = self._controller_colors
        side = self.left_or_right
        color = cc[side] if side in cc else cc['default']
        from numpy import array, uint8
        rgba8 = array(color, uint8)
        return rgba8

    @property
    def room_position(self):
        hm = self.hand_model
        return hm.room_position if hm else None

    @property
    def tip_room_position(self):
        hm = self.hand_model
        return hm.room_position.origin() if hm else None

    @property
    def position(self):
        hm = self.hand_model
        return hm.position if hm else None

    @property
    def button_modes(self):
        return self._modes

    @property
    def left_or_right(self):
        return self._side
    
    def close(self):
        self._device_index = None
        self._active_drag_modes.clear()
        self._close_hand_model()

    def _close_hand_model(self):
        hm = self._hand_model
        if hm:
            if not hm.deleted:
                hm.delete()
            self._hand_model = None
        
    def _update_position(self):
        '''Move hand controller model to new position.
        Keep size constant in physical room units.'''
        di = self._device_index
        if di is None:
            return

        hm = self.hand_model
        if hm is None:
            # Hand model was delete by user, so recreate it.
            hm = self._create_hand_model()

        hpos = self._camera.device_position(di)
        if hpos is not None:
            hm.room_position = hpos
        self.update_scene_position()

    def update_scene_position(self):
        hm = self.hand_model
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
        hm = self.hand_model
        if hm:
            hm._show_button_down(b, pressed)
        m = self._modes.get(b)
        if not isinstance(m, ShowUIMode):
            # Check for click on UI panel.
            ui = self._camera.user_interface
            if ui.process_hand_controller_button_event(self, b, pressed, released):
                return
        
        # Call HandMode event callback.
        if m:
            event = HandButtonEvent(self, b, pressed = pressed, released = released)
            if b == openvr.k_EButton_SteamVR_Touchpad:
                x, y = self._touchpad_position()
                if x is not None and y is not None:
                    event.touchpad_position = (x,y)
            self._dispatch_event(m, event)

    def _dispatch_event(self, mode, hand_event):
        if hand_event.pressed:
            mode.pressed(hand_event)
            mode._button_down = hand_event.button		# Used for detecting missing button release events
            self._active_drag_modes.add(mode)
        elif hand_event.released:
            mode.released(hand_event)
            self._active_drag_modes.discard(mode)
            self._update_ui(mode)

    def _update_ui(self, mode):
        if mode.update_ui_on_release:
            f = mode.update_ui_delay_frames
            self._camera.user_interface.redraw_ui(delay_frames = f)

    def current_hand_mode(self, button):
        return self._modes.get(button)
    
    def set_hand_mode(self, button, hand_mode):
        self._modes[button] = hand_mode
        hm = self.hand_model
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
                m.drag(HandMotionEvent(self, m._button_down, previous_pose, pose))

        # Check for Oculus thumbstick position
        self._send_thumbstick_events()

    def _send_thumbstick_events(self):
        ts_mode = self._thumbstick_mode()
        if ts_mode is None or not ts_mode.uses_thumbstick():
            return
        x,y = self._thumbstick_position()
        if x is None or y is None:
            return
        min_tilt = .1
        if abs(x) < min_tilt and abs(y) < min_tilt:
            return

        event = HandThumbstickEvent(self, x, y)
        ts_mode.thumbstick(event)
        
        if event.took_step:
            self._update_ui(ts_mode)

    def _thumbstick_position(self):
        # Position range is -1 to 1 on each axis.
        success, cstate = self._vr_system.getControllerState(self._device_index)
        if success:
            # On Oculus Rift S, axis 0=thumbstick, 1=trigger, 2=grip
            astate = cstate.rAxis[0]
            return astate.x, astate.y
        return None, None

    def _touchpad_position(self):
        return self._thumbstick_position()

    def _check_for_missing_button_release(self):
        '''Cancel drag modes if button has been released even if we didn't get a button up event.'''
        adm = self._active_drag_modes
        if len(adm) == 0:
            return
        success, cstate = self._vr_system.getControllerState(self._device_index)
        if success:
            pressed_mask = cstate.ulButtonPressed
            for m in tuple(adm):
                b = m._button_down
                # bm = openvr.ButtonMaskFromId(b)  # Routine is missing from pyopenvr
                bm = 1 << b
                if not pressed_mask & bm:
                    self._dispatch_event(m, HandButtonEvent(self, b, released = True))

from chimerax.core.models import Model
class HandModel(Model):
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False

    def __init__(self, session, name, length = 0.20, radius = 0.04, color = (200,200,0,255),
                 controller_type = 'htc vive'):
        Model.__init__(self, name, session)

        from chimerax.geometry import Place
        self.room_position = Place()	# Hand controller position in room coordinates.

        self._cone_color = color
        self._button_color = (255,255,255,255)	# White
        self.color = (255,255,255,255)	# Texture modulation color

        self._controller_type = controller_type
        
        # Avoid hand disappearing when behind models, especially in multiperson VR.
        self.allow_depth_cue = False

        # Don't let clip planes hide hand models.
        self.allow_clipping = False
        
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
        from chimerax.graphics import concatenate_geometry
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

    def draw(self, renderer, draw_pass):
        if not self._hide_hand():
            Model.draw(self, renderer, draw_pass)

    def _hide_hand(self):
        '''
        Returns true if the room camera is currently being rendered
        and the hand models are to be hidden.
        '''
        c = self.session.main_view.camera
        if isinstance(c, SteamVRCamera):
            rc = c.room_camera
            hide = (rc and rc.is_rendering and not rc.show_hands)
            return hide
        return False

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
        from chimerax.graphics import Texture
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
            from Qt.QtGui import QImage
            qi = QImage(icon_path)
            s = image_size
            if qi.width() != s or qi.height() != s:
                qi = qi.scaled(s,s)
            from chimerax.graphics import qimage_to_numpy
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

class HandEvent:
    def __init__(self, hand_controller, button):
        self._hand_controller = hand_controller
        self._button = button
        self._touchpad_position = (None,None)
    @property
    def hand_controller(self):
        return self._hand_controller
    @property
    def button(self):
        return self._button
    @property
    def is_touchpad(self):
        import openvr
        return self._button == openvr.k_EButton_SteamVR_Touchpad
    def _get_touchpad_position(self):
        return self._touchpad_position
    def _set_touchpad_position(self, xy):
        self._touchpad_position = tuple(xy)
    touchpad_position = property(_get_touchpad_position, _set_touchpad_position)
    @property
    def camera(self):
        return self.hand_controller._camera
    @property
    def position(self):
        '''Scene coordinates Place.'''
        rp = self.hand_controller.room_position
        rts = self.camera.room_to_scene
        p = rts * rp
        return p
    @property
    def tip_position(self):
        '''Scene coordinates point.'''
        return self.position.origin()
    def picking_segment(self):
        '''Range is given in scene units.'''
        p = self.hand_controller.position
        xyz1 = p * (0,0,0)
        xyz2 = p * (0,0,-self._picking_range)
        return xyz1, xyz2
    @property
    def _picking_range(self):
        return self.camera.user_interface._mouse_mode_click_range

class HandButtonEvent(HandEvent):
    def __init__(self, hand_controller, button, pressed = False, released = False):
        HandEvent.__init__(self, hand_controller, button)
        self._pressed = pressed
        self._released = released
    @property
    def pressed(self):
        return self._pressed
    @property
    def released(self):
        return self._released
    def picked_object(self, view):
        '''Return pick for object pointed at, along ray from cone.'''
        xyz1, xyz2 = self.picking_segment()
        pick = view.picked_object_on_segment(xyz1, xyz2)
        return pick
    
class HandMotionEvent(HandEvent):
    def __init__(self, hand_controller, button, previous_pose, current_pose):
        HandEvent.__init__(self, hand_controller, button)
        self._previous_pose = previous_pose
        self._current_pose = current_pose
        self._last_drag_room_position = None	# May be from earlier than previous HandMotionEvent
    @property
    def pose(self):
        '''Room coordinates Place.'''
        return self._current_pose
    @property
    def previous_pose(self):
        '''Room coordinates Place.'''
        return self._previous_pose
    @property
    def motion(self):
        '''Rotation and translation in scene coordinates give as a Place instance.'''
        rp = self.hand_controller.room_position
        ldp = self._last_drag_room_position
        if ldp is None:
            from chimerax.geometry import Place
            move = Place()
        else:
            room_move = rp * ldp.inverse()
            rts = self.camera.room_to_scene
            move = rts * room_move * rts.inverse()
        return move
    def set_last_drag_position(self, last_position):
        self._last_drag_room_position = last_position
    @property
    def room_vertical_motion(self):
        '''In meters.'''
        rp = self.hand_controller.room_position
        ldp = self._last_drag_room_position
        delta_z = (rp.origin() - ldp.origin())[1] # Room vertical motion
        return delta_z
    @property
    def tip_motion(self):
        '''Scene coordinates vector.'''
        p = self.tip_position
        return self.motion * p - p

class HandThumbstickEvent(HandEvent):
    def __init__(self, hand_controller, x, y):
        import openvr
        button = openvr.k_EButton_SteamVR_Touchpad
        HandEvent.__init__(self, hand_controller, button)
        self._x = x
        self._y = y

        # Settings for Oculus thumbstick used as single-step control
        self._thumbstick_release_level = 0.2
        self._thumbstick_click_level = 0.5
        self._thumbstick_repeat_delay = 0.3  # seconds
        self._thumbstick_repeat_interval = 0.1  # seconds
        if not hasattr(hand_controller, '_thumbstick_state'):
            hand_controller._thumbstick_state = {
                'released': False, 'repeating': False, 'time': None}

        self.took_step = False
        
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y

    def thumbstick_step(self, flip_y = False):
        '''Return 1,0,-1 when thumbstick tilted.'''
        x,y = self.x, self.y
        ts = self.hand_controller._thumbstick_state
        if not ts['released']:
            release = self._thumbstick_release_level
            if abs(x) < release and abs(y) < release:
                ts['released'] = True
                ts['repeating'] = False
            elif ts['time'] is not None:
                repeat = self._thumbstick_repeat_interval if ts['repeating'] else self._thumbstick_repeat_delay
                from time import time
                if time() - ts['time'] > repeat:
                    ts['released'] = True
                    ts['repeating'] = True
        if not ts['released']:
            return 0
        click = self._thumbstick_click_level
        if abs(x) < click and abs(y) < click:
            return 0
        ts['released'] = False
        from time import time
        ts['time'] = time()
        v = x if abs(x) > abs(y) else (-y if flip_y else y)
        step = 1 if v > 0 else -1
        if step:
            self.took_step = True
        return step
        
class HandMode:
    name = 'unnamed'
    update_ui_on_release = True
    update_ui_delay_frames = None
    @property
    def icon_path(self):
        return None
    def pressed(self, hand_button_event):
        pass
    def released(self, hand_button_event):
        pass
    def drag(self, hand_motion_event):
        pass
    uses_touch_motion = False
    def touch(self):
        pass
    def untouch(self):
        pass
    def uses_thumbstick(self):
        return False
    def thumbstick(self, hand_thumbstick_event):
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
    def pressed(self, hand_event):
        hc = hand_event.hand_controller
        self._last_hand_position[hc] = hc.room_position
    def released(self, hand_event):
        self._last_hand_position[hand_event.hand_controller] = None
        self._panel = None
    def drag(self, hand_motion_event):
        e = hand_motion_event
        ui = e.camera.user_interface
        hc = e.hand_controller
        oc = e.camera.other_controller(hc)
        if oc and self._ui_zoom(oc):
            scale, center = _pinch_scale(e.previous_pose.origin(), e.pose.origin(), oc.tip_room_position)
            scale = max(min(scale, 10.0), 0.1)	# Limit scaling
            ui.scale_ui(scale)
            self._last_hand_position.clear()	# Avoid jump when one button released
        else:
            hrp = hc.room_position
            lhrp = self._last_hand_position.get(hc)
            if lhrp is not None:
                room_motion = hrp * lhrp.inverse()
                panel = self._panel
                if panel is None:
                    ui.move(room_motion)
                else:
                    panel.move(room_motion)
                self._last_hand_position[hc] = hrp
    def _ui_zoom(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, MoveUIMode):
                return True
        return False

class ShowUIMode(MoveUIMode):
    name = 'show ui'
    update_ui_on_release = False
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
    def pressed(self, hand_event):
        c = hand_event.camera
        ui = c.user_interface
        if ui.shown():
            from time import time
            self._start_ui_move_time = time()
        else:
            ui.display_ui(hand_event.hand_controller.room_position, c.room_position)
        MoveUIMode.pressed(self, hand_event)
    def released(self, hand_event):
        # End UI move, or hide.
        stime = self._start_ui_move_time
        from time import time
        if stime is not None and time() < stime + self._ui_hide_time:
            hand_event.camera.user_interface.hide()
        self._start_ui_move_time = None
        MoveUIMode.released(self, hand_event)

class MoveSceneMode(HandMode):
    name = 'move scene'
    names = ('move scene', 'rotate', 'translate')
    update_ui_on_release = False
    def __init__(self):
        self._zoom_center = None
    @property
    def icon_path(self):
        return MoveSceneMode.icon_location()

    @staticmethod
    def icon_location():
        from chimerax.mouse_modes import TranslateMouseMode
        return TranslateMouseMode.icon_location()

    def drag(self, hand_motion_event):
        e = hand_motion_event
        cam = e.camera
        oc = cam.other_controller(e.hand_controller)
        if oc and self._other_controller_move(oc):
            # Both controllers trying to move scene -- zoom
            scale, center = _pinch_scale(e.previous_pose.origin(), e.pose.origin(), oc.tip_room_position)
            if self._zoom_center is None:
                self._zoom_center = _choose_zoom_center(cam, center)
            _pinch_zoom(cam, scale, self._zoom_center)
        else:
            self._zoom_center = None
            move = e.pose * e.previous_pose.inverse()
            cam.move_scene(move)

    def released(self, hand_event):
        self._zoom_center = None
        
    def _other_controller_move(self, oc):
        for m in oc._active_drag_modes:
            if isinstance(m, MoveSceneMode):
                return True
        return False

    def uses_thumbstick(self):
        return True

    def thumbstick(self, hand_thumbstick_event):
        e = hand_thumbstick_event
        x,y = e.x, e.y
        from math import sqrt
        r = sqrt(x*x + y*y)
        limit = .2
        if r < limit:
            return

        f = (r-limit)/(1-limit)
        angle = f*f	# degrees
        camera = e.camera
        center = _choose_zoom_center(camera)
        (vx,vy,vz) = center - camera.room_position.origin()
        from chimerax.geometry import normalize_vector, rotation
        horz_dir = normalize_vector((vz,0,-vx))
        from numpy import array, float32
        vert_dir = array((0,1,0),float32)  # y-axis is up in room coordinates
        axis = normalize_vector(x*vert_dir + y*horz_dir)
        move = rotation(axis, angle, center)
        camera.move_scene(move)

def _pinch_scale(prev_pos, pos, other_pos):
    from chimerax.geometry import distance
    d, dp = distance(pos,other_pos), distance(prev_pos,other_pos)
    if dp > 0:
        s = d / dp
    else:
        s = 1.0
    center = 0.5*(pos+other_pos)
        
    return s, center

class ZoomMode(HandMode):
    name = 'zoom'
    update_ui_on_release = False
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
    def pressed(self, hand_event):
        tip_position = hand_event.hand_controller.tip_room_position
        self._zoom_center = _choose_zoom_center(hand_event.camera, tip_position)
    def drag(self, hand_motion_event):
        e = hand_motion_event
        center = self._zoom_center
        if center is None:
            return
        y_motion = (e.pose.origin() - e.previous_pose.origin())[1]  # meters
        scale_factor = 2 ** (y_motion/self.size_doubling_distance)
        _pinch_zoom(e.camera, scale_factor, center)
    def released(self, hand_event):
        self._zoom_center = None
    def uses_thumbstick(self):
        return True
    def thumbstick(self, hand_thumbstick_event):
        e = hand_thumbstick_event
        y  = e.y
        limit = .2
        if abs(y) > limit:
            center = _choose_zoom_center(e.camera, e.hand_controller.tip_room_position)
            v = (y-limit if y > 0 else y+limit)/(1-limit)
            scale_factor = 1.0 - 0.02 * v * abs(v)
            _pinch_zoom(e.camera, scale_factor, center)

def _choose_zoom_center(camera, center = None):
    # Zoom in about center of scene if requested center point is outside scene bounding box.
    # This avoids pushing a distant scene away.
    b = camera.vr_view.drawing_bounds()
    if b and (center is None or not b.contains_point(camera.room_to_scene * center)):
        return camera.room_to_scene.inverse() * b.center()
    if center is None:
        return camera.room_position.origin()
    return center

def _pinch_zoom(camera, scale_factor, center, max_scale_factor = 10, max_scale = 1e12):
    if max_scale_factor is not None:
        if scale_factor > max_scale_factor:
            scale_factor = max_scale_factor
        elif scale_factor < 1/max_scale_factor:
            scale_factor = 1/max_scale_factor
    if max_scale is not None:
        s = camera.scene_scale * scale_factor
        if s > max_scale or s < 1/max_scale:
            return
    from chimerax.geometry import distance, translation, scale
    scale = translation(center) * scale(scale_factor) * translation(-center)
    camera.move_scene(scale)

class RecenterMode(HandMode):
    name = 'recenter'
    def pressed(self, hand_event):
        hand_event.camera.fit_scene_to_room()
    @property
    def icon_path(self):
        return self.icon_location()
    @staticmethod
    def icon_location():
        from os.path import join, dirname
        from chimerax import shortcuts
        return join(dirname(shortcuts.__file__), 'icons', 'viewall.png')

class MouseMode(HandMode):
    def __init__(self, mouse_mode):
        self._mouse_mode = mouse_mode
        mouse_mode.enable()
        self.name = mouse_mode.name
        self._last_drag_room_position = None # Hand controller position at last vr_motion call
        if hasattr(mouse_mode, 'vr_update_delay_frames'):
            # Some modes need longer to update GUI after a click, like ViewDockX.
            self.update_ui_delay_frames = mouse_mode.vr_update_delay_frames

    @property
    def has_vr_support(self):
        m = self._mouse_mode
        return (hasattr(m, 'vr_press') or
                hasattr(m, 'vr_motion') or
                hasattr(m, 'vr_release') or
                hasattr(m, 'vr_thumbstick'))

    @property
    def icon_path(self):
        return self._mouse_mode.icon_path
    
    def pressed(self, hand_button_event):
        self._click(hand_button_event)

    def released(self, hand_button_event):
        self._click(hand_button_event)

    def _click(self, hand_button_event):
        e = hand_button_event
        m = self._mouse_mode
        if e.pressed and hasattr(m, 'vr_press'):
            m.vr_press(e)
        if hasattr(m, 'vr_motion'):
            self._last_drag_room_position = e.hand_controller.room_position if e.pressed else None
        if e.released and hasattr(m, 'vr_release'):
            m.vr_release(e)

    def drag(self, hand_motion_event):
        m = self._mouse_mode
        if not hasattr(m, 'vr_motion'):
            return
        
        e = hand_motion_event
        e.set_last_drag_position(self._last_drag_room_position)
        if m.vr_motion(e) != 'accumulate drag':
            self._last_drag_room_position = e.hand_controller.room_position

    def uses_thumbstick(self):
        return hasattr(self._mouse_mode, 'vr_thumbstick')
    
    def thumbstick(self, hand_thumbstick_event):
        '''Generate a mouse mode wheel event when thumbstick pushed.'''
        if self.uses_thumbstick():
            self._mouse_mode.vr_thumbstick(hand_thumbstick_event)

class RunCommandMode(HandMode):
    name = 'command'
    update_ui_on_release = True
    def __init__(self, session, command):
        self._session = session
        self._command = command
        self.name = 'command "%s"' % command
    @property
    def icon_path(self):
        return RunCommandMode.icon_location()
    @staticmethod
    def icon_location():
        from os.path import join, dirname
        return join(dirname(__file__), 'command_icon.png')
    def pressed(self, hand_event):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        try:
            run(self._session, self._command)
        except UserError as e:
            self._session.logger.warning(str(e))
        except Exception as e:
            from traceback import format_exc
            self._session.logger.bug(format_exc())
            
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
    from chimerax.geometry import Place
    from numpy import array, float32
    p = Place(array(hmat34.m, float32))
    return p
    
