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
def vr(session, enable = None, room_position = None, mirror = False):
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
      Whether to update the ChimeraX graphics window.  This will usually cause judder
      in the vr headset because the computer display is running at a refresh rate of 60
      frames per second and will slow the rendering to the headset.  (May be able to turn off
      syncing to vertical refresh to avoid this.)  It is better to use the SteamVR display
      mirror window.
    '''
    
    if enable is None and room_position is None:
        enable = True

    if enable is not None:
        if enable:
            start_vr(session)
        else:
            stop_vr(session)

    if room_position is not None:
        c = session.main_view.camera
        if not isinstance(c, SteamVRCamera):
            from chimerax.core.errors import UserError
            raise UserError('Cannot use vr roomPosition unless vr enabled.')
        if isinstance(room_position, str) and room_position == 'report':
            p = ','.join('%.5g' % x for x in tuple(c.room_to_scene.matrix.flat))
            session.logger.info(p)
        else:
            c.room_to_scene = room_position
            c._last_position = c.position

    if mirror is not None:
        c = session.main_view.camera
        if isinstance(c, SteamVRCamera):
            c.mirror_display = mirror
            
        
# -----------------------------------------------------------------------------
# Register the oculus command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('room_position', Or(EnumOf(['report']), PlaceArg)),
                              ('mirror', BoolArg)],
                   synopsis = 'Start SteamVR virtual reality rendering')
    register('device vr', desc, vr, logger=logger)
    create_alias('vr', 'device vr $*', logger=logger)

# -----------------------------------------------------------------------------
#
def start_vr(session):

    v = session.main_view
    if isinstance(v.camera, SteamVRCamera):
        return

    try:
        import openvr
    except Exception as e:
        from chimerax.core.errors import UserError
        raise UserError('Failed to import OpenVR module: %s' % str(e))
    
    v.camera = SteamVRCamera(session)
    # Set redraw timer for 1 msec to minimize dropped frames.
    session.ui.main_window.graphics_window.set_redraw_interval(1)

    msg = 'started SteamVR rendering'
    log = session.logger
    log.status(msg)
    log.info(msg)

# -----------------------------------------------------------------------------
#
def stop_vr(session):

    v = session.main_view
    c = v.camera
    if isinstance(c, SteamVRCamera):
        # Have to delay shutdown of SteamVR connection until draw callback
        # otherwise it clobbers the Qt OpenGL context making entire gui black.
        def replace_camera(s = session):
            from chimerax.core.graphics import MonoCamera
            s.main_view.camera = MonoCamera()
            s.ui.main_window.graphics_window.set_redraw_interval(10)
        c.close(replace_camera)
    

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
        self._controller_poses = {}	# Controller device pose while trigger pressed
        self._close = False
        self._controller_models = {}	# Device id to HandControllerModel
        self._controller_colors = ((200,200,0,255), (0,200,200,255))
        self._trigger_held = {}		# Map device id to bool
        self._move_selected_atoms = {}	# Map device id to bool
        self.mirror_display = False	# Mirror right eye in ChimeraX window
        				# This causes stuttering in the Vive.
        
        import openvr
        self.vr_system = vrs = openvr.init(openvr.VRApplication_Scene)
        self._controller_ids = [d for d in range(openvr.k_unMaxTrackedDeviceCount)
                                if vrs.getTrackedDeviceClass(d) == openvr.TrackedDeviceClass_Controller]

        self._render_size = self.vr_system.getRecommendedRenderTargetSize()
        self.compositor = openvr.VRCompositor()
        if self.compositor is None:
            raise RuntimeError("Unable to create compositor") 

        # Compute projection and eye matrices, units in meters
        zNear = 0.1
        zFar = 500.0
        # TODO: Scaling models to be huge causes clipping at far clip plane.

        # Left and right projections are different. OpenGL 4x4.
        pl = vrs.getProjectionMatrix(openvr.Eye_Left, zNear, zFar, openvr.API_OpenGL)
        self.projection_left = hmd44_to_opengl44(pl)
        pr = vrs.getProjectionMatrix(openvr.Eye_Right, zNear, zFar, openvr.API_OpenGL)
        self.projection_right = hmd44_to_opengl44(pr)

        # Eye shifts from hmd pose.
        vl = vrs.getEyeToHeadTransform(openvr.Eye_Left)
        self.eye_shift_left = hmd34_to_position(vl)
        vr = vrs.getEyeToHeadTransform(openvr.Eye_Right)
        self.eye_shift_right = hmd34_to_position(vr)

        # Map ChimeraX scene coordinates to OpenVR room coordinates
        bounds = session.main_view.drawing_bounds()
        self.fit_scene_to_room(bounds)
        
        # Update camera position every frame.
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self._poses = poses_t()
        h = session.triggers.add_handler('new frame', self.next_frame)
        self._new_frame_handler = h

    def fit_scene_to_room(self,
                          scene_bounds,
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

    def close(self, close_cb = None):
        cm = [m for d,m in self._controller_models.items()]
        if cm:
            self._session.models.close(cm)
            self._controller_models = {}
        self._close = True
        self._close_cb = close_cb
        
    def _delayed_close(self):
        # Apparently OpenVR doesn't make its OpenGL context current
        # before deleting resources.  If the Qt GUI opengl context is current
        # openvr deletes the Qt resources instead.  So delay openvr close
        # until after rendering so that openvr opengl context is current.
        self._session.triggers.remove_handler(self._new_frame_handler)
        self._new_frame_handler = None
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
        Cnew = self.room_to_scene * H * S
        self.position = self._last_position = Cnew
        self._last_h = H

    @property
    def scene_scale(self):
        '''Scale factor from scene to room coordinates.'''
        x,y,z = self.room_to_scene.matrix[:,0]
        from math import sqrt
        return 1/sqrt(x*x + y*y + z*z)
    
    def process_controller_events(self):

        self.process_controller_buttons()
        self.process_controller_motion()
        self.move_controller_models()

    def process_controller_buttons(self):
        
        # Check for button press
        vrs = self.vr_system
        import openvr
        e = openvr.VREvent_t()
        if not vrs.pollNextEvent(e):
            return
        
        t = e.eventType
        if t == openvr.VREvent_ButtonPress or t == openvr.VREvent_ButtonUnpress:
            pressed = (t == openvr.VREvent_ButtonPress)
            d = e.trackedDeviceIndex
            b = e.data.controller.button
            if b == openvr.k_EButton_SteamVR_Trigger:
                self._trigger_held[d] = pressed
#            press = 'press' if pressed else 'unpress'
#            print('Controller button %s, device %d, button %d'
#                      % (press, d, b))
            elif b == openvr.k_EButton_SteamVR_Touchpad:
                if pressed:
                    hm = self.hand_controller_models()
                    m = hm.get(d, None)
                    if m:
                        a = m.closest_atom()
                        if a:
                            # Select atom with bottom of touchpad,
                            # or residue with top of touchpad
                            xy = self.touchpad_position(d)
                            if xy is not None:
                                self._session.selection.clear()
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
            elif b == openvr.k_EButton_ApplicationMenu:
                pass
            elif b == openvr.k_EButton_Grip:
                self._move_selected_atoms[d] = pressed

    def touchpad_position(self, device_id):
        vrs = self.vr_system
        from ctypes import sizeof
        # TODO: I think pyopenvr eliminated the size arg in Feb 2017.
        import openvr
        size = sizeof(openvr.VRControllerState_t)
        success, cs = vrs.getControllerState(device_id, size)
        if success:
            a = cs.rAxis[0]
            return (a.x, a.y)
        return None

    def process_controller_motion(self):

        # For controllers with trigger pressed, use controller motion to move scene
        # Rotation and scaling is about controller position -- has natural feel,
        # like you grab the models where your hand is located.
        # Another idea is to instead pretend controller is at center of models.
        acm = self.controller_motions()
        cm = [acm[d] for d,h in self._trigger_held.items() if h]
        if len(cm) == 1:
            # One controller has trigger pressed, move scene.
            previous_pose, pose = cm[0]
            move = previous_pose * pose.inverse()
            self.room_to_scene = self.room_to_scene * move
        elif len(cm) == 2:
            # Two controllers have trigger pressed, scale scene.
            (prev_pose1, pose1), (prev_pose2, pose2) = cm
            pp1, p1 = prev_pose1.origin(), pose1.origin()
            pp2, p2 = prev_pose2.origin(), pose2.origin()
            from chimerax.core.geometry import distance, translation, scale
            d, dp = distance(p1,p2), distance(pp1,pp2)
            center = 0.5*(p1+p2)
            if d > 0.5*dp:
                s = dp / d
                scale = translation(center) * scale(s) * translation(-center)
                self.room_to_scene = self.room_to_scene * scale

        # Move selected atoms
        mam = [acm[d] for d,m in self._move_selected_atoms.items() if m]
        if len(mam) == 1:
            previous_pose, pose = mam[0]
            move = pose * previous_pose.inverse()  # Room to room coords
            rts = self.room_to_scene
            smove = rts * move * rts.inverse()	# Scene to scene coords.
            from chimerax.core.atomic import selected_atoms
            atoms = selected_atoms(self._session)
            atoms.scene_coords = smove * atoms.scene_coords

    def controller_motions(self):
        '''Return list of (pose, previous_pose) for controllers with trigger pressed.'''
        cm = {}
        cp = self._controller_poses
        for d in self._controller_ids:
            # Pose maps controller to room coords.
            pose = hmd34_to_position(self._poses[d].mDeviceToAbsoluteTracking)
            previous_pose = cp.get(d, None)
            cp[d] = pose
            if previous_pose:
                cm[d] = (previous_pose, pose)
        return cm

    def controller_poses(self):
        poses = {d:hmd34_to_position(self._poses[d].mDeviceToAbsoluteTracking)
                 for d in self._controller_ids}
        return poses

    def move_controller_models(self):
        hm = self.hand_controller_models()
        cp = self.controller_poses()
        for d in self._controller_ids:
                hm[d].update_position(cp[d], self.room_to_scene)
        
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

    def view_all(self, bounds, aspect = None, pad = 0):
        fov = 100	# Effective field of view, degrees
        from chimerax.core.graphics.camera import perspective_view_all
        self.position = perspective_view_all(bounds, self.position, fov, aspect, pad)
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
            # Submit left eye texture (view 0) before rendering right eye (view 1)
            import openvr
            self.compositor.submit(openvr.Eye_Left, fb.openvr_texture)

    def combine_rendered_camera_views(self, render):
        '''
        Submit right eye texture image to OpenVR. Left eye was already submitted
        by set_render_target() when render target switched to right eye.
        '''
        fb = render.pop_framebuffer()
        import openvr
        self.compositor.submit(openvr.Eye_Right, fb.openvr_texture)

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
            ovrt.eType = openvr.API_OpenGL
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

    def hand_controller_models(self):
        hm = self._controller_models
        if len(hm) == 0:
            ses = self._session
            cc = self._controller_colors
            hmlist = []
            for i, d in enumerate(self._controller_ids):
                hm[d] = HandControllerModel('Hand %s' % (i+1), ses, cc[i%len(cc)])
                hmlist.append(hm[d])
            ses.models.add(hmlist)
        return hm


from chimerax.core.models import Model
class HandControllerModel(Model):

    def __init__(self, name, session, rgba8, size = 0.20, aspect = 0.2):
        Model.__init__(self, name, session)
        from chimerax.core.surface.shapes import cone_geometry
        va, na, ta = cone_geometry(nc = 50, points_up = False)
        va[:,:2] *= aspect
        va[:,2] += 0.5		# Move tip to 0,0,0 for picking
        va *= size
        self.geometry = va, ta
        self.normals = na
        from numpy import array, uint8
        self.color = array(rgba8, uint8)

    def update_position(self, room_place, room_to_scene):
        '''Move hand controller to new position.
        Keep size constant in physical room units.'''
        self.position = room_to_scene * room_place

    def tip_position(self):
        return self.scene_position.origin()
    
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
    
