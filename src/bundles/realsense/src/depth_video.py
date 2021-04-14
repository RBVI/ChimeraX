# vim: set expandtab ts=4 sw=4:

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
#
def device_realsense(session, enable = True,
                     size = (960,540), dsize = (1280,720), frames_per_second = 30,
                     align = True, denoise = True, denoise_weight = 0.1,
                     denoise_color_tolerance = 10, projector = False,
                     angstroms_per_meter = 50, skip_frames = 2, set_window_size = True):

    di = session.models.list(type = DepthVideo)
    if enable:
        if di:
            session.models.close(di)

        di = DepthVideo('RealSense camera', session,
                        size = size,
                        dsize = dsize,
                        frames_per_second = frames_per_second,
                        depth_scale = angstroms_per_meter,
                        use_ir_projector = projector,
                        align_color_and_depth = align,
                        denoise_depth = denoise,
                        denoise_weight = denoise_weight,
                        denoise_color_tolerance = denoise_color_tolerance,
                        skip_frames = skip_frames)
        session.models.add([di])
        msg = 'RealSense camera model #%s' % di.id_string
        session.logger.info(msg)
        if set_window_size:
            di.set_window_size()
    else:
        session.models.close(di)
            
# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import BoolArg, FloatArg, IntArg, Int2Arg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('size', Int2Arg),
                              ('dsize', Int2Arg),
                              ('frames_per_second', IntArg),
                              ('align', BoolArg),
                              ('denoise', BoolArg),
                              ('denoise_weight', FloatArg),
                              ('denoise_color_tolerance', IntArg),
                              ('projector', BoolArg),
                              ('angstroms_per_meter', FloatArg),
                              ('skip_frames', IntArg),
                              ('set_window_size', BoolArg),
                   ],
                   synopsis = 'Turn on RealSense camera rendering',
                   url = 'help:user/commands/device.html#realsense')
    register('realsense', desc, device_realsense, logger=logger)
    create_alias('device realsense', 'realsense $*', logger=logger,
                 url='help:user/commands/device.html#realsense')
            
# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class DepthVideo (Model):
    skip_bounds = True
    SESSION_SAVE = False
    def __init__(self, name, session,
                 size = (960,540),		# color frame size in pixels
                 dsize = (1280,720),		# depth frame size in pixels
                 frames_per_second = 30,
                 skip_frames = 0,
                 depth_scale = 50,	        # Angstroms per meter.
                 use_ir_projector = False,      # Interferes with Vive VR tracking
                 align_color_and_depth = True,  # This slows frame rate
                 denoise_depth = True,		# Average depth values
                 denoise_weight = 0.1,		# Weight factor for current frame depth
                 denoise_color_tolerance = 10	# Color matching used by denoising
    ):
        Model.__init__(self, name, session)

        # With VR camera depth scale is taken from camera.
        self._depth_scale = depth_scale		# Angstroms per meter.

        self._first_image = True
        self._render_field_of_view = 69.4	# TODO: Get this from chimerax camera
        self._realsense_color_field_of_view = (69.4,42.5) # TODO: Get this from pyrealsense
        self._realsense_depth_field_of_view = (91.2,65.5) # TODO: Get this from pyrealsense
        self._use_ir_projector = use_ir_projector
        self._align_color_and_depth = align_color_and_depth
        self._denoise_depth = denoise_depth
        self._denoise_weight = denoise_weight
        self._denoise_color_tolerance = denoise_color_tolerance
        self._pipeline_started = False
        self._color_image_size = size
        self._depth_image_size = dsize
        # RealSense D435 frame rate:
        #  30, 15, 6 at depth 1280x720, or 60,90 at 848x480
        #  6,15,30 at color 1920x1080, 60 at 1280x720
        self._frames_per_second = frames_per_second
        self._skip_frames = skip_frames	# Skip updating realsense on some graphics updates
        self._current_frame = 0
        self._depth_texture = None
        
        t = session.triggers.add_handler('graphics update', self._update_image)
        self._update_trigger = t

        self._start_video()

    def delete(self):
        t = self._update_trigger
        if t:
            self.session.triggers.remove_handler(t)
            self._update_trigger = None

        p = self.pipeline
        if p:
            if self._pipeline_started:
                p.stop()
            self.pipeline = None
            
        Model.delete(self)

        # Do this after Model.delete() so opengl context made current
        dt = self._depth_texture
        if dt:
            dt.delete_texture()
            self._depth_texture = None

    def set_window_size(self):
        w,h = self._color_image_size
        from chimerax.core.commands import run
        run(self.session, 'windowsize %d %d' % (w,h))
        
    def _start_video(self):
        # Configure depth and color streams
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        self.config = config = rs.config()
        fps = self._frames_per_second
        dw, dh = self._depth_image_size
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
        cw, ch = self._color_image_size
        config.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, fps)
        pipeline_profile = self.pipeline.start(config)
        device = pipeline_profile.get_device()
        dsensor = device.first_depth_sensor()
        if dsensor.supports(rs.option.emitter_enabled):
            enable = 1 if self._use_ir_projector else 0
            dsensor.set_option(rs.option.emitter_enabled, enable) # Turn on/off IR projector
        self._pipeline_started = True

        # Setup aligning depth images to color images
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def _update_image(self, tname, view):

        cam = view.camera
        if hasattr(cam, 'have_room_camera') and not cam.have_room_camera:
            cmd = 'vr roomCamera on fieldOfView %.1f' % self._render_field_of_view
            if hasattr(cam, 'have_tracker') and cam.have_tracker:
                cmd += ' tracker on'
            from chimerax.core.commands import run
            run(self.session, cmd)
            
        if not self.display:
            if self._pipeline_started:
                # Stop video processing if not displayed.
                self.pipeline.stop()
                self._pipeline_started = False
            return False
        elif not self._pipeline_started:
            # Restart video processing when displayed.
            self.pipeline.start(self.config)
            self._pipeline_started = True

        skip = self._skip_frames
        if skip > 0:
            self._current_frame += 1
            if self._current_frame % (skip+1) != 1:
                return

        import pyrealsense2 as rs
        '''
        frames = rs.composite_frame(rs.frame())
        if self.pipeline.poll_for_frames(frames) == 0:
            return
        '''
        frames = self.pipeline.poll_for_frames()
        if frames.size() != 2:  # Got depth and color stream
            return

        
        # Wait for a coherent pair of frames: depth and color
        # This blocks if frames not available.
        # frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        # TODO: Alignment is slow causing stuttering in VR.  Interpolate aligned depth
        #       values on GPU by scaling texture coordinate.
        if self._align_color_and_depth:
            aligned_frames = self.align.process(frames)
        else:
            aligned_frames = frames

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

#        if not self._first_image:
#            return	# Test if texture load is slowing VR

        import numpy
        # Convert images to numpy arrays
        depth_image = numpy.asanyarray(depth_frame.get_data())
        color_image = numpy.asanyarray(color_frame.get_data())
#        if view.frame_number % 100 == -1:
#            print ('depth', depth_image.dtype, depth_image.shape,
#                   'color', color_image.dtype, color_image.shape)
#            print ('depth values')
#            for r in range(10):
#                print(' '.join('%5d' % d for d in depth_image[230+r, 310:320]))

        if self._first_image:
            self._create_textures_video(color_image, depth_image)
            self._first_image = False
            ci = color_frame.profile.as_video_stream_profile().intrinsics
#            print('color intrinsics', ci)
            cfov = rs.rs2_fov(ci)
#            print('color fov', cfov)
            self._realsense_color_field_of_view = cfov
            di = depth_frame.profile.as_video_stream_profile().intrinsics
#            print('depth intrinsics', di)
            dfov = rs.rs2_fov(di)
#            print('depth fov', dfov)
            self._realsense_depth_field_of_view = dfov
#            print('extrinsics color to depth', color_frame.profile.get_extrinsics_to(depth_frame.profile))
            if self._denoise_depth:
                self._ave_depth_image = depth_image.copy()
                self._max_depth = depth_image.copy()
                self._max_depth_color = color_image.copy()
                self._last_color_image = color_image.copy()
                # from numpy import int16
                # self._last_color_image = color_image.copy().astype(int16)
        else:
            self.texture.reload_texture(color_image)

            if self._denoise_depth:
                ave_depth = self._ave_depth_image
                # Make sure _depthvideo can runtime link shared library libarrays.
                from chimerax import arrays ; arrays.load_libarrays()
                from ._depthvideo import denoise_depth
                denoise_depth(depth_image, color_image,
                              ave_depth = ave_depth, ave_weight = self._denoise_weight,
                              max_depth = self._max_depth, max_depth_color = self._max_depth_color,
                              last_color = self._last_color_image,
                              max_color_diff = self._denoise_color_tolerance)
                self._depth_texture.reload_texture(ave_depth)
                '''
                # Average depth over several frames to reduce flicker
                ave = self._ave_depth_image
                f = .1
                from numpy import average, putmask, abs, logical_or
                putmask(depth_image, (depth_image == 0), ave)  # depth 0 values are unknown, don't change average
                ave[:] = average((depth_image, ave), axis = 0, weights = (f, 1-f)).astype(ave.dtype)

                # Update depth without averaging if color change a lot.
                cdiff = abs(color_image - self._last_color_image)
                cmax = 10
                cfast = (cdiff[:,:,0] > cmax)
                logical_or(cdiff[:,:,1] > cmax, cfast, cfast)
                logical_or(cdiff[:,:,2] > cmax, cfast, cfast)
                putmask(ave, cfast, depth_image)
                self._last_color_image[:] = color_image

                self._depth_texture.reload_texture(ave)
                '''
            else:
                self._depth_texture.reload_texture(depth_image)

        self.redraw_needed()

    def _create_textures_video(self, color_image, depth_image):
        # TODO: Does not have sensible bounds.  Bounds don't really make sense.
        #       Causes surprises if it is the first model opened.
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, color_image, (-1, -1), (2, 2))
        # Invert y-axis by flipping texture coordinates
        self.texture_coordinates[:,1] = 1 - self.texture_coordinates[:,1]
        from chimerax.graphics import Texture
        self._depth_texture = dt = Texture(depth_image)
        # Shader wants to handle 0 depth values (= unknown depth) as max distance
        # so need to turn off linear interpolation so fragment shader gets 0 values.
        dt.linear_interpolation = False
#        print ('color image type', color_image.dtype, color_image.shape)
#        print ('depth image type', depth_image.dtype, depth_image.shape,
#               'mean', depth_image.mean(), 'min', depth_image.min(), 'max', depth_image.max())

    def _create_textures_test(self):
        w = h = 512
        w1,h1,w2,h2 = w//4, h//4, 3*w//4, 3*h//4
        from numpy import empty, uint8, float32, uint16
        color = empty((h,w,4), uint8)
        color[:] = 255
        color[h1:h2,w1:w2,0] = 0
        #    depth = empty((h,w), float32)
        #    depth[:] = 0.5
        #    depth[h1:h2,w1:w2] = 1.0
        depth = empty((h,w), uint16)
        depth[:] = 32000
        depth[h1:h2,w1:w2] = 64000
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, color, (-1, -1), (2, 2))
        from chimerax.graphics import Texture
        self._depth_texture = Texture(depth)
        
    def draw(self, renderer, draw_pass):
        '''Render a color and depth texture pair.'''
        if self._first_image:
            return
        if not getattr(renderer, 'mix_video', True):
            return
        draw = ((draw_pass == self.OPAQUE_DRAW_PASS and self.opaque_texture)
                or (draw_pass == self.TRANSPARENT_DRAW_PASS and not self.opaque_texture))
        if not draw:
            return

        r = renderer
        r.disable_shader_capabilities(r.SHADER_LIGHTING |
                                      r.SHADER_STEREO_360 |	# Avoid geometry shift
                                      r.SHADER_DEPTH_CUE |
                                      r.SHADER_SHADOW |
                                      r.SHADER_MULTISHADOW |
                                      r.SHADER_CLIP_PLANES)
        r.enable_capabilities |= r.SHADER_DEPTH_TEXTURE

        # If the desired field of view of the texture does not match the camera field of view
        # then adjust projection size.  Also if the apect ratio of the target framebuffer and
        # the aspect ratio of the texture don't match adjust the projection size.
        w,h = r.render_size()
        fx = self._render_field_of_view
        rsfx, rsfy = self._realsense_color_field_of_view
        from math import atan, radians
        rw = atan(radians(fx/2))
        rh = rw*h/w
        rsw, rsh = atan(radians(rsfx/2)), atan(radians(rsfy/2))
        sx, sy =  rsw/rw, rsh/rh
        
        cur_proj = r.current_projection_matrix
        r.set_projection_matrix(((sx, 0, 0, 0), (0, sy, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        from chimerax.geometry import place
        p0 = place.identity()
        cur_view = r.current_view_matrix
        r.set_view_matrix(p0)
        r.set_model_matrix(p0)

        t = self._depth_texture
        t.bind_texture(r.depth_texture_unit)
        frm = 2**16 / 1000  # Realsense full range in meters (~65).
        c = self.session.main_view.camera
        from chimerax.vive.vr import SteamVRCamera
        if isinstance(c, SteamVRCamera):
            # Scale factor from RealSense depth texture 0-1 range
            # (~65 meters) to scene units (typically Angstroms).
            depth_scale = frm / c.scene_scale
        else:
            depth_scale = self._depth_scale * frm
        from math import tan, radians
        cxfov, cyfov = self._realsense_color_field_of_view
        dxfov, dyfov = self._realsense_depth_field_of_view
        dxscale = tan(radians(0.5*cxfov)) / tan(radians(0.5*dxfov))
        dyscale = tan(radians(0.5*cyfov)) / tan(radians(0.5*dyfov))
        r.set_depth_texture_parameters(dxscale, dyscale, depth_scale)
#        if r.frame_number % 200 == 1:
#            print ('depth params', dxscale, dyscale, depth_scale)
        Model.draw(self, r, draw_pass)

        # Restore view and projection matrices since drawings are not supposed to change these.
        r.set_projection_matrix(cur_proj)
        r.set_view_matrix(cur_view)
        
        r.enable_capabilities &= ~r.SHADER_DEPTH_TEXTURE
        r.disable_shader_capabilities(0)
