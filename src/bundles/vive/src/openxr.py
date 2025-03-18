# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

class XR:
    '''
    Encapsulate all OpenXR API calls.

    Copied most of the OpenXR setup code from xr_examples/gl_example.py in
        https://github.com/cmbruns/pyopenxr_examples
    '''
    def __init__(self):
        self.field_of_view = [None, None]		# Field of view for each eye, radians
        self.eye_pose = [None, None]			# Last eye camera pose
        self._debug = True
        
        self._instance = None				# Connection to OpenXR runtime
        self._system_id = None				# Headset hardware
        self._session = None				# Connection to headset
        self._session_state = None
        self._projection_layer = None
        self._swapchains = []				# Render textures
        self._framebuffers = []				# Framebuffers for left and right eyes
        self.render_size = (1000,1000)
        self._ready_to_render = False
        self._scene_space = None			# XrSpace instance reference coordinate space
        self._hand_space = {}				# XrSpace instances for left and right hands
        self._passthrough_supported = False
        self._passthrough = None

        self._frame_started = False
        self._frame_count = 0

        self._action_set = None
        self._event_queue = []

    def start_session(self):
        self._instance = self._create_instance()	# Connect to OpenXR runtime
        self._system_id = self._create_system()		# Choose headset
        self._session = self._create_session()		# Connect to headset
        self._projection_layer = self._create_projection_layer()
        self.render_size = self._recommended_render_size()
        self._frame_started = False
        self._frame_count = 0
        self._action_set = self._create_action_set()	# Manage hand controller input

    def _create_instance(self):
        '''Establish connection to OpenXR runtime.'''
        import xr
        requested_extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]

        from . import passthrough
        self._passthrough_supported = passthrough.passthrough_extension_available()
        if self._passthrough_supported:
            requested_extensions.append(xr.FB_PASSTHROUGH_EXTENSION_NAME)
            
            
        if self._debug:
            requested_extensions.append(xr.EXT_DEBUG_UTILS_EXTENSION_NAME)

        app_info = xr.ApplicationInfo("chimerax", 0, "pyopenxr", 0, xr.XR_API_VERSION_1_0)
        iinfo = xr.InstanceCreateInfo(application_info = app_info,
                                      enabled_extension_names = requested_extensions)
        if self._debug:
            self._debug_cb = _enable_openxr_debugging(iinfo)

        instance = xr.create_instance(iinfo)

        return instance

    def enable_passthrough_video(self, enable):
        if enable == 'toggle':
            enable = (self._passthrough is None)

        import xr
        if enable and self._passthrough is None:
            if not self._passthrough_supported:
                from chimerax.core.errors import UserError
                raise UserError('The Facebook passthrough video extension is not supported by this instance of OpenXR')
            from .passthrough import Passthrough
            self._passthrough = Passthrough(self._session, self._instance)
            print ('Enabled passthrough')
        elif not enable and self._passthrough is not None:
            self._passthrough.close()
            self._passthrough = None

    def runtime_name(self):
        '''
        Oculus runtime reports "Oculus"
        SteamVR runtime reports "SteamVR/OpenXR".
        '''
        import xr
        props = xr.get_instance_properties(instance=self._instance)
        rt_name = props.runtime_name.decode('utf-8')
        return rt_name
    
    def _create_system(self):
        '''Find headset.'''
        import xr
        get_info = xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        try:
            system_id = xr.get_system(self._instance, get_info)
        except xr.exception.FormFactorUnavailableError:
            raise RuntimeError(self._headset_not_found_message())
        return system_id

    def _headset_not_found_message(self):
         rt = self.runtime_name()
         msg = 'Could not find VR headset.\n\n'
         msg += f'The active OpenXR runtime is {rt}.\n\n'
         if 'Oculus' in rt:
             msg += ('Perhaps the Oculus application is not running, '
                     'or the headset is not connected by Quest Link (cable) or Air Link (wifi). '
                     'Check in the "Devices" section of the Oculus application to see if the headset is connected. '
                     'If you want instead to use SteamVR then press "Set SteamVR as OpenXR Runtime" '
                     'in the SteamVR application under Settings / OpenXR.')
         elif 'SteamVR' in rt:
             msg += ('Perhaps the SteamVR application is not running, '
                     'or the headset is not detected (e.g. Vive link box turned off).'
                     'If you want instead to use Oculus then press "Set Oculus as Active" '
                     'in the Oculus application under Settings / General / OpenXR Runtime.')
         else:
             msg += ('Perhaps the Oculus or SteamVR application is not running, '
                     'or the headset is not detected, or you have not set the OpenXR runtime to use. '
                     'If you want to use Oculus then press "Set Oculus as Active" '
                     'in the Oculus application under Settings / General / OpenXR Runtime. '
                     'If you want to use SteamVR then press "Set SteamVR as OpenXR Runtime" '
                     'in the SteamVR application under Settings / OpenXR.')
         return msg

    def system_name(self):
        '''
        Oculus runtime with Quest 2 reports "Oculus Quest 2"
        SteamVR runtime with Valve Index headset reports "SteamVR/OpenXR: lighthouse".
        SteamVR runtime with Oculus headset reports "SteamVR/OpenXR: oculus".
        '''
        import xr
        props = xr.get_system_properties(instance=self._instance,
                                         system_id=self._system_id)
        sys_name = props.system_name.decode('utf-8')
        return sys_name

    def device_properties(self):
        '''
        Sony Spatial Reality SR1 15.6" display vendor id = 195951310.
        Might need this to distinguish from SR2 27" model.
        '''
        import xr
        props = xr.get_system_properties(instance=self._instance,
                                         system_id=self._system_id)
        tprops = props.tracking_properties
        return {'system_name': props.system_name.decode('utf-8'),
                'vendor_id': props.vendor_id,
                'position_tracking': tprops.position_tracking,
                'orientation_tracking': tprops.orientation_tracking,
                }

    def _recommended_render_size(self):
        '''Width and height of single eye framebuffer.'''
        import xr
        view_configs = xr.enumerate_view_configurations(self._instance, self._system_id)
        assert view_configs[0] == xr.ViewConfigurationType.PRIMARY_STEREO.value
        view_config_views = xr.enumerate_view_configuration_views(
            self._instance, self._system_id, xr.ViewConfigurationType.PRIMARY_STEREO)
        assert len(view_config_views) == 2
        view0 = view_config_views[0]
        return (view0.recommended_image_rect_width, view0.recommended_image_rect_height)

    def _create_session(self):
        '''
        Connect to headset.
        This requires the OpenGL context to be current because it passes the graphics
        context as an argument when creating the session.
        '''
        self._get_graphics_requirements()
        import xr
        gb = xr.GraphicsBindingOpenGLWin32KHR()
        from OpenGL import WGL
        gb.h_dc = WGL.wglGetCurrentDC()
        gb.h_glrc = WGL.wglGetCurrentContext()
        debug ('WGL dc', gb.h_dc, 'WGL context', gb.h_glrc)
        self._graphics_binding = gb
        import ctypes
        pp = ctypes.cast(ctypes.pointer(gb), ctypes.c_void_p)
        sesinfo = xr.SessionCreateInfo(0, self._system_id, next=pp)
        session = xr.create_session(self._instance, sesinfo)
        space = xr.create_reference_space(session,
                                          xr.ReferenceSpaceCreateInfo(xr.ReferenceSpaceType.STAGE))
        self._scene_space = space
        return session

    def _get_graphics_requirements(self):
        # Have to request graphics requirements before xrCreateSession()
        # otherwise OpenXR generates an error.
        import ctypes, xr
        pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(
                self._instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        result = pxrGetOpenGLGraphicsRequirementsKHR(
            self._instance, self._system_id,
            ctypes.byref(graphics_requirements))
        result = xr.exception.check_result(xr.Result(result))
        if result.is_exception():
            raise result

        min_ver = graphics_requirements.min_api_version_supported
        max_ver = graphics_requirements.max_api_version_supported
        debug (f'OpenXR requires OpenGL version min {min_ver}, max {max_ver}')

    def _create_projection_layer(self):
        '''Set projection mode'''
        import xr
        pl_views = (xr.CompositionLayerProjectionView * 2)(
            xr.CompositionLayerProjectionView(), xr.CompositionLayerProjectionView())
        self._projection_layer_views = pl_views
        # Set layer_flags to allow passthrough to be composited correctly.
        if self._passthrough_supported:
            flags = (xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                     xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT)
        else:
            flags = 0
        layer_flags = xr.CompositionLayerFlags(flags)
        projection_layer = xr.CompositionLayerProjection(layer_flags = layer_flags,
                                                         space = self._scene_space,
                                                         views = pl_views)
        return projection_layer

    def _create_swapchains(self):
        '''Make render textures'''
        from OpenGL import GL
        #
        # OpenXR requires linear color space read from its swapchain
        # textures.  That means either an OpenGL SRGB texture is used
        # that automaically converts from internal SRGB to linear when
        # sampled.  Or a non-SRGB texture is used and the texture values
        # must be in linear color space.  Since ChimeraX never uses
        # linear color space we need to use an SRGB texture.  And
        # it appears GL_FRAMEBUFFER_SRGB is disabled so writing to
        # this texture is not doing any conversion.
        #
        # This leaves the problem that when we render the desktop
        # mirror image using the SRGB texture it converts to linear
        # automatically and that gives the wrong colors.  How do I
        # fix this?  There is an obscure SKIP_DECODE opengl texture
        # parameter but that might break the OpenXR rendering.
        # Another choice is to convert linear to sRGB in our fragment
        # shader when sampling the SRGB texture to draw the desktop
        # mirror framebuffer.  We would need a special fragment shader
        # option to enable that encoding.
        #
        # Currently the SRGB format gives VR colors similar to screen
        # when not in VR, but the mirror shows darker colors.  With
        # non-SRGB formats the VR shows brighter paler colors while
        # the mirror shows correct colors.  Results are the same whether
        # using the SteamVR or Oculus OpenXR runtimes.
        #
        color_formats = [GL.GL_SRGB8_ALPHA8]
        color_formats.append(GL.GL_RGBA16F)  # Works on SteamVR with Vive Pro (881A)
        # Supported openxr color formats on Oculus Quest 2. 8058, 881B, 8C3A, 8C43
        color_formats.extend([GL.GL_RGBA8, GL.GL_RGB16F, GL.GL_R11F_G11F_B10F, GL.GL_SRGB8_ALPHA8])
        import xr
        swapchain_formats = xr.enumerate_swapchain_formats(self._session)
        for color_format in color_formats:
            if color_format in swapchain_formats:
                break
        if color_format not in swapchain_formats:
            format_nums = ['%0X' % fmt for fmt in swapchain_formats]
            raise ValueError(f'Color format {color_format} not in supported formats {format_nums}')
        w,h = self.render_size
        scinfo = xr.SwapchainCreateInfo(
            usage_flags = (xr.SwapchainUsageFlags.SAMPLED_BIT |
                           xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT),
            #usage_flags = xr.SWAPCHAIN_USAGE_TRANSFER_DST_BIT,
            format = color_format,
            sample_count = 1,
            array_size = 1,
            face_count = 1,
            mip_count = 1,
            width = w,
            height = h)
        
        swapchains = [xr.create_swapchain(self._session, scinfo),	# Left and right eye
                      xr.create_swapchain(self._session, scinfo)]

        # Keep the buffer alive by moving it into the list of buffers.
        self._swapchain_image_buffer = [
            xr.enumerate_swapchain_images(swapchain=swapchain,
                                          element_type=xr.SwapchainImageOpenGLKHR)
            for swapchain in swapchains]

        for sc, plv in zip(swapchains, self._projection_layer_views):
            plv.sub_image.swapchain = sc
            plv.sub_image.image_rect.offset[:] = (0,0)
            plv.sub_image.image_rect.extent[:] = (w, h)

        return swapchains

    @property
    def framebuffers(self):
        return self._framebuffers
    
    def _create_framebuffers(self, opengl_context):
        if len(self._swapchains) == 0:
            self._swapchains = self._create_swapchains()
        w,h = self.render_size
        from chimerax.graphics.opengl import Texture, Framebuffer
        framebuffers = []
        for sc, eye in zip(self._swapchains, ('left','right')):
            import xr
            images = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)
            tex_id = images[0].image
            # TODO: This is a misuse of the Texture class.  It is supposed to allocate
            #   the texture and it will delete it when the Texture is deleted.
            t = Texture()
            t.id = tex_id
            t.size = (w,h)
            fb = Framebuffer(f'VR {eye} eye', opengl_context,
                             width=w, height=h, color_texture = t, alpha = True)
            framebuffers.append(fb)
        return framebuffers

    def _delete_framebuffers(self):
        for fb in self._framebuffers:
            fb.color_texture.id = None  # Avoid deleting the swapchain texture.
            fb.delete(make_current = True)
        self._framebuffers.clear()

    def set_opengl_render_target(self, render, eye):
        if not self._frame_started:
            return
        if len(self._framebuffers) == 0:
            self._framebuffers = self._create_framebuffers(render.opengl_context)
        ei = 0 if eye == 'left' else 1
        swapchain = self._swapchains[ei]
        import xr
        swapchain_index = xr.acquire_swapchain_image(swapchain=swapchain,
                                                     acquire_info=xr.SwapchainImageAcquireInfo())
        xr.wait_swapchain_image(swapchain=swapchain,
                                wait_info=xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
        fb = self._framebuffers[ei]
        render.push_framebuffer(fb)
        images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
        tex_id = images[swapchain_index].image
        fb.color_texture.id = tex_id
        from chimerax.graphics.opengl import GL
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, tex_id, 0)
        # The swap chains consist of 3 images, probably to pipeline the rendering,
        # so each frame the next image is used.
        debug('render target', eye, 'texture', tex_id)
        
    def release_opengl_render_target(self, render, eye):
        if not self._frame_started:
            return
        render.pop_framebuffer()
        ei = 0 if eye == 'left' else 1
        swapchain = self._swapchains[ei]
        import xr
        xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())

    def start_frame(self):
        self._frame_started = False
        if not self._ready_to_render:
            return False
        self._poll_xr_events()
        if self._start_xr_frame():
            debug ('started xr frame')
            if self._update_xr_views() and self._frame_state.should_render:
                self._frame_count += 1
                debug ('frame should render')
                self._frame_started = True
                return True
            else:
                self._end_xr_frame()
        return False

    def end_frame(self):
        if self._frame_started:
            self._frame_started = False
            self._end_xr_frame()
            debug('ended xr frame')

    def _poll_xr_events(self):
        import xr
        while True:
            if self.connection_closed:
                return
            try:
                event_buffer = xr.poll_event(self._instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError:
                    # PyOpenXR gives ValueError for event types it does not know
                    # such as from the XR_FB_passthrough extension.
                    continue
                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    self._on_session_state_changed(event_buffer)
                else:
                    debug('Got event', event_type)
            except xr.EventUnavailable:
                break

    def _on_session_state_changed(self, session_state_changed_event):
        import xr
        import ctypes
        event = ctypes.cast(
            ctypes.byref(session_state_changed_event),
            ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
        self._session_state = state = xr.SessionState(event.state)
        debug('Session state', state)
        if state == xr.SessionState.READY:
            self._begin_session()
        elif state == xr.SessionState.STOPPING:
            self._end_session()
            # After this it will transition to the IDLE state and from there
            # it will either go back to READY or EXITING.
            # No calls should be made to wait_frame, begin_frame, end_frame until ready.
        elif state == xr.SessionState.EXITING:
            self.shutdown()

    def _begin_session(self):
        import xr
        sbinfo = xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO)
        xr.begin_session(self._session, sbinfo)
        self._ready_to_render = True

    def _end_session(self):
        import xr
        xr.end_session(self._session)
        self._ready_to_render = False
            
    def _start_xr_frame(self):
        import xr
        if self._session_state in [
            xr.SessionState.READY,
            xr.SessionState.FOCUSED,
            xr.SessionState.SYNCHRONIZED,
            xr.SessionState.VISIBLE,
        ]:
            try:
                self._frame_state = xr.wait_frame(self._session,
                                                  frame_wait_info=xr.FrameWaitInfo())
            except xr.ResultException as e:
                if not getattr(self, '_last_start_frame_failed', False):
                    error (f'xr.wait_frame() failed: {e}')
                self._last_start_frame_failed = True
                return False
            try:
                xr.begin_frame(self._session, xr.FrameBeginInfo())
            except xr.ResultException as e:
                if not getattr(self, '_last_start_frame_failed', False):
                    error (f'xr.begin_frame() failed: {e}')
                    if self.system_name() == 'SonySRD System':
                        # On the Sony turning off the display power or letting
                        # it sleep too long will cause begin_frame() to fail
                        # after which I found no way to revive the OpenXR session.
                        # ChimeraX ticket #.
                        error('The Sony Spatial Reality display appears to be turned off or sleeping.  Unfortunately the Sony OpenXR driver is broken and if the Sony display slept you will need to turn OpenXR off and back on in ChimeraX to get it to work again.  If you attempted to start OpenXR when the Sony display is off, you will need to restart ChimeraX to get it to work.')
                self._last_start_frame_failed = True
                return False
            self._last_start_frame_failed = False
            return True
                
        return False

    def _end_xr_frame(self):
        layers = []
        import xr
        blend_mode = xr.EnvironmentBlendMode.OPAQUE
        if self._frame_state.should_render and hasattr(self, '_eye_view_states'):
            for eye_index in range(2):
                layer_view = self._projection_layer_views[eye_index]
                eye_view = self._eye_view_states[eye_index]
                layer_view.fov = eye_view.fov
                layer_view.pose = eye_view.pose
            import ctypes
            layers = [ctypes.byref(self._projection_layer)]
            if self._passthrough is not None:
                pclayer = self._passthrough._composition_layer
                layers.insert(0, ctypes.byref(pclayer))

        self._projection_layer.views = self._projection_layer_views

        frame_end_info = xr.FrameEndInfo(
            display_time = self._frame_state.predicted_display_time,
            environment_blend_mode = blend_mode,
            layers = layers)
        xr.end_frame(self._session, frame_end_info)

    def _update_xr_views(self):
        import xr
        vs, evs = xr.locate_views(self._session,
                                  xr.ViewLocateInfo(
                                      view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                                      display_time=self._frame_state.predicted_display_time,
                                      space=self._projection_layer.space))
        vsf = vs.view_state_flags
        if (vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0 or
            vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0):
            return False  # There are no valid tracking poses for the views.

        if len(evs) != 2:
            return False	# Acer stereo display can return fewer than 2 views.  Bug #16151

        self._eye_view_states = evs
        for eye_index, view_state in enumerate(evs):
            self.field_of_view[eye_index] = view_state.fov
            self.eye_pose[eye_index] = self._xr_pose_to_place(view_state.pose)

        return True
    
    def _xr_pose_to_place(self, xr_pose):
        from chimerax.geometry import quaternion_rotation
        x,y,z,w = xr_pose.orientation
        q = (w,x,y,z)
        return quaternion_rotation(q, xr_pose.position)

    def _create_action_set(self):
        import xr
        # Create an action set.
        action_set_info = xr.ActionSetCreateInfo(
            action_set_name="gameplay",
            localized_action_set_name="Gameplay",
            priority=0,
        )
        action_set = xr.create_action_set(self._instance, action_set_info)

        self._hand_path = {'left': self._xr_path("/user/hand/left"),
                           'right': self._xr_path("/user/hand/right")}

        self._actions = self._create_button_actions(action_set)
        
        pose_bindings = self._create_hand_pose_actions(action_set)
        khr_bindings = self._suggested_bindings([
            ('/user/hand/left/input/select/click', 'trigger'),
            ('/user/hand/right/input/select/click', 'trigger'),
            ('/user/hand/left/input/menu/click', 'menu'),
            ('/user/hand/right/input/menu/click', 'menu'),
        ])
        self._suggest_bindings(pose_bindings + khr_bindings,
                               "/interaction_profiles/khr/simple_controller")

        vive_bindings = self._suggested_bindings([
            ('/user/hand/left/input/trigger/click','trigger'),
            ('/user/hand/right/input/trigger/click','trigger'),
            ('/user/hand/left/input/squeeze/click', 'grip'),
            ('/user/hand/right/input/squeeze/click', 'grip'),
            ('/user/hand/left/input/menu/click', 'menu'),
            ('/user/hand/right/input/menu/click', 'menu'),
            ('/user/hand/left/input/trackpad/click', 'touchpad'),
            ('/user/hand/right/input/trackpad/click', 'touchpad'),
        ])
        self._suggest_bindings(pose_bindings + vive_bindings,
                               "/interaction_profiles/htc/vive_controller")

        oculus_bindings = self._suggested_bindings([
            ('/user/hand/left/input/trigger/value', 'trigger_value'),
            ('/user/hand/right/input/trigger/value', 'trigger_value'),
            ('/user/hand/left/input/squeeze/value', 'grip_value'),
            ('/user/hand/right/input/squeeze/value', 'grip_value'),
            ('/user/hand/left/input/thumbstick', 'thumbstick'),
            ('/user/hand/right/input/thumbstick', 'thumbstick'),
            ('/user/hand/left/input/thumbstick/click', 'thumbstick_press'),
            ('/user/hand/right/input/thumbstick/click', 'thumbstick_press'),
            ('/user/hand/left/input/x/click', 'A'),
            ('/user/hand/right/input/a/click', 'A'),
            ('/user/hand/left/input/y/click', 'menu'),
            ('/user/hand/right/input/b/click', 'menu'),
        ])
        self._suggest_bindings(pose_bindings + oculus_bindings,
                               "/interaction_profiles/oculus/touch_controller")

        from ctypes import pointer
        xr.attach_session_action_sets(
            session=self._session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=1,
                action_sets=pointer(action_set),
            ),
        )
        return action_set

    def _create_hand_pose_actions(self, action_set):
        # Get the XrPath for the left and right hands.
        import xr

        # Create an input action getting the left and right hand poses.
        self._hand_pose_action = xr.create_action(
            action_set=action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose",
                localized_action_name="Hand Pose",
                count_subaction_paths=2,
                subaction_paths=[self._hand_path['left'], self._hand_path['right']],
            ),
        )

        # Suggest bindings
        bindings = [
            xr.ActionSuggestedBinding(self._hand_pose_action,
                                      self._xr_path("/user/hand/left/input/aim/pose")),
            xr.ActionSuggestedBinding(self._hand_pose_action,
                                      self._xr_path("/user/hand/right/input/aim/pose"))
        ]

        # Create action space
        left_action_space_info = xr.ActionSpaceCreateInfo(
            action=self._hand_pose_action,
            subaction_path=self._hand_path['left'],
        )
        assert left_action_space_info.pose_in_action_space.orientation.w == 1
        right_action_space_info = xr.ActionSpaceCreateInfo(
            action=self._hand_pose_action,
            subaction_path=self._hand_path['right'],
        )
        assert right_action_space_info.pose_in_action_space.orientation.w == 1
        self._hand_space = {
            'left' : xr.create_action_space(session=self._session,
                                            create_info=left_action_space_info),
            'right': xr.create_action_space(session=self._session,
                                            create_info=right_action_space_info)
        }

        return bindings

    def _xr_path(self, path):
        import xr
        return xr.string_to_path(self._instance, path)
    
    def _create_button_actions(self, action_set):
        actions = {}
        import xr
        both_hands = ['left', 'right']
        from xr import ActionType as t
        for button, hands, type in (
                ('trigger', both_hands, t.BOOLEAN_INPUT),
                ('trigger_value', both_hands, t.FLOAT_INPUT),
                ('grip', both_hands, t.BOOLEAN_INPUT),
                ('grip_value', both_hands, t.FLOAT_INPUT),
                ('menu', both_hands, t.BOOLEAN_INPUT),
                ('touchpad', both_hands, t.BOOLEAN_INPUT),
                ('A', both_hands, t.BOOLEAN_INPUT),
                ('thumbstick', both_hands, t.VECTOR2F_INPUT),
                ('thumbstick_press', both_hands, t.BOOLEAN_INPUT),
        ):
            action = xr.create_action(
                action_set=action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=type,
                    action_name=button.lower(),
                    localized_action_name=button,
                    count_subaction_paths=len(hands),
                    subaction_paths=[self._hand_path[side] for side in hands],
                ),
            )
            actions[button] = Action(action, button, hands, type, self._hand_path)
        
        return actions

    def _suggested_bindings(self, path_to_action_names):
        bindings = []
        import xr
        for path, action_name in path_to_action_names:
            ca = self._actions[action_name]
            bindings.append(xr.ActionSuggestedBinding(ca.action, self._xr_path(path)))
        return bindings
    
    def _suggest_bindings(self, bindings, profile_path):
        import xr
        xr.suggest_interaction_profile_bindings(
            instance=self._instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(self._instance, profile_path),
                count_suggested_bindings=len(bindings),
                suggested_bindings=(xr.ActionSuggestedBinding * len(bindings))(*bindings),
            ),
        )

    def device_active(self, device_name):
        '''device_name is "left" or "right"'''
        import xr
        if self._session_state != xr.SessionState.FOCUSED:
            return False

        pose_state = xr.get_action_state_pose(
            session=self._session,
            get_info=xr.ActionStateGetInfo(
                action=self._hand_pose_action,
                subaction_path=self._hand_path[device_name],
            ),
        )

        return pose_state.is_active

    def device_position(self, device_name):
        '''device_name is "left" or "right"'''
        if not self.device_active(device_name):
            return None

        import xr
        space_location = xr.locate_space(
            space=self._hand_space[device_name],
            base_space=self._scene_space,
            time=self._frame_state.predicted_display_time,
        )
        loc_flags = space_location.location_flags
        if (loc_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT == 0
            or loc_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT == 0):
            return None

        return self._xr_pose_to_place(space_location.pose)

    def controller_model_name(self, device_name):
        runtime = self.runtime_name()
        sysname = self.system_name()
        if runtime.startswith('Oculus') or sysname.endswith('oculus'):
            model_name = f'oculus {device_name}'
        elif runtime.startswith('SteamVR') and sysname.endswith('lighthouse'):
            model_name = 'htc vive'
        else:
            model_name = 'unknown'
        return model_name
        
    def shutdown(self):
        import xr
        for hs in self._hand_space.values():
            xr.destroy_space(hs)
        self._hand_space.clear()

        if self._action_set is not None:
            xr.destroy_action_set(self._action_set)
            self._action_set = None
        
        self._delete_framebuffers()

        # Delete projection layer, swapchains, session, system and instance.
        for swapchain in self._swapchains:
            xr.destroy_swapchain(swapchain)
        self._swapchains = []

#        self._projection_layer.destroy()  # Destroy method is missing
        self._projection_layer = None

        if self._scene_space is not None:
            xr.destroy_space(self._scene_space)
            self._scene_space = None

        if self._session is not None:
            if self.system_name() == 'SonySRD System':
                # Without these two calls OpenXR crashes in destroy_session()
                # if a Sony Spatial Reality display disconnects when it is powered off.
                xr.request_exit_session(self._session)
                xr.end_session(self._session)
            xr.destroy_session(self._session)
            self._session = None

        if self._instance is not None:
            xr.destroy_instance(self._instance)
            self._instance = None

    @property
    def connection_closed(self):
        return self._instance is None
    
    def headset_pose(self):
        # head to room coordinates.  None if not available
        e0, e1 = self.eye_pose
        if e0 is None or e1 is None:
            return None
        shift = 0.5 * (e1.origin() - e0.origin())
        from chimerax.geometry import translation
        return translation(shift) * e0

    def poll_next_event(self):
        self._poll_xr_events()	# Update self._session_state to detect headset has lost focus
        if self.connection_closed:
            return None
        q = self._event_queue
        q.extend(self._button_events())
        if len(q) == 0:
            return None
        e = q[0]
        del q[0]
        return e

    def _button_events(self):
        if not self._sync_actions():
            return []
        
        events = []
        for a in self._actions.values():
            events.extend(a.events(self._session))
                
        return events

    def _sync_actions(self):
        import xr
        if self._session_state != xr.SessionState.FOCUSED:
            return False
        active_action_set = xr.ActiveActionSet(self._action_set, xr.NULL_PATH)
        from ctypes import pointer
        xr.sync_actions(
            self._session,
            xr.ActionsSyncInfo(
                count_active_action_sets=1,
                active_action_sets=pointer(active_action_set)
            ),
        )
        return True

    def hand_controllers(self):
        # Return list of (device_id, 'left' or 'right')
        return []
    def controller_left_or_right(self, device_index):
        # Return 'left' or 'right'
        return 'right'
    def controller_state(self, device_name):
        return None
    def device_type(self, device_index):
        # Returns 'controller', 'tracker', 'hmd', or 'unknown'
        return 'unknown'
    def find_tracker(self):
        # Return device id for connected "tracker" device or None
        return None
    TrackedDeviceActivated = 0
    TrackedDeviceDeactivated = 1
    ButtonTouchEvent = 4
    ButtonUntouchEvent = 5

class Action:
    def __init__(self, action, button, sides, type, hand_paths):
        self.action = action
        self.button = button
        self.sides = sides
        self.type = type	# xr.ActionType.BOOLEAN_INPUT or FLOAT_INPUT or VECTOR2F_INPUT
        self._hand_path = hand_paths

        self._float_press = 0.7
        self._float_release = 0.3
        self._float_state = {'left':0, 'right':0}

        self._xy_minimum = 0.1		# Minimum thumbstick value to generate an event

    @property
    def button_name(self):
        b = self.button
        if b.endswith('_value'):
            return b[:-6]
        elif b.endswith('_press'):
            return b[:-6]
        return b
    
    def events(self, session):
        import xr
        if self.type == xr.ActionType.BOOLEAN_INPUT:
            return self._bool_events(session)
        elif self.type == xr.ActionType.FLOAT_INPUT:
            return self._float_events(session)
        elif self.type == xr.ActionType.VECTOR2F_INPUT:
            return self._xy_events(session)

    def _bool_events(self, session):
        events = []
        import xr
        for side in self.sides:
            b_state = xr.get_action_state_boolean(
                session=session,
                get_info=xr.ActionStateGetInfo(action = self.action,
                                               subaction_path=self._hand_path[side]))
            if b_state.is_active and b_state.changed_since_last_sync:
                state = 'pressed' if b_state.current_state else 'released'
                events.append(ButtonEvent(self.button_name, state, side))
        return events

    def _float_events(self, session):
        events = []
        import xr
        for side in self.sides:
            f_state = xr.get_action_state_float(
                session=session,
                get_info=xr.ActionStateGetInfo(action = self.action,
                                               subaction_path=self._hand_path[side]))
            if f_state.is_active and f_state.changed_since_last_sync:
                value = f_state.current_state
                if value > self._float_press:
                    if self._float_state[side] < self._float_press:
                        events.append(ButtonEvent(self.button_name, 'pressed', side))
                elif value < self._float_release:
                    if self._float_state[side] > self._float_release:
                        events.append(ButtonEvent(self.button_name, 'released', side))
                self._float_state[side] = value
        return events

    def _xy_events(self, session):
        events = []
        import xr
        for side in self.sides:
            xy_state = xr.get_action_state_vector2f(
                session=session,
                get_info=xr.ActionStateGetInfo(action = self.action,
                                               subaction_path=self._hand_path[side]))
            if xy_state.is_active and xy_state.changed_since_last_sync:
                value = xy_state.current_state
                if abs(value.x) >= self._xy_minimum or abs(value.y) >= self._xy_minimum:
                    xy_event = XYEvent(self.button_name, (value.x,value.y), side)
                    events.append(xy_event)
        return events
    
class ButtonEvent:
    def __init__(self, button_name, state, device_name):
        self.button_name = button_name
        self.state = state	# "pressed" or "released"
        self.device_name = device_name
    
class XYEvent:
    def __init__(self, button_name, xy, device_name):
        self.button_name = button_name
        self.xy = xy
        self.device_name = device_name

def _enable_openxr_debugging(iinfo):
    import xr
    dumci = xr.DebugUtilsMessengerCreateInfoEXT()
    dumci.message_severities = (xr.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                                | xr.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                                | xr.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | xr.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    dumci.message_types = (xr.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                           | xr.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                           | xr.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                           | xr.DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT)
    dumci.user_data = None
    _debug_cb = xr.PFN_xrDebugUtilsMessengerCallbackEXT(_debug_callback)
    dumci.user_callback = _debug_cb
    import ctypes
    iinfo.next = ctypes.cast(ctypes.pointer(dumci), ctypes.c_void_p)
    return _debug_cb  # Need to keep a Python reference to this so it is not deleted.

def _debug_callback(severity, _type, data, _user_data):
    d = data.contents
    sev = _debug_severity_string(severity)
    fname = d.function_name.decode() if d.function_name is not None else "unknown func"
    msg = d.message.decode()
    message = f"{sev}: {fname}: {msg}"
    if sev in ('Error', 'Warning'):
        error(message)
    else:
        debug(message)
    return True

def _debug_severity_string(severity_flags):
    if severity_flags & 0x0001:
        return 'Verbose'
    if severity_flags & 0x0010:
        return 'Info'
    if severity_flags & 0x0100:
        return 'Warning'
    if severity_flags & 0x1000:
        return 'Error'
    return 'Critical'

def error(*args):
    print(*args)
    
def debug(*args):
    pass
#    print(*args)
