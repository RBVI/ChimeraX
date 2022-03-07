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
# Qt 6 video capture.
#
from chimerax.core.models import Model
class WebCam (Model):
    skip_bounds = True
    casts_shadows = False
    SESSION_SAVE = False
    def __init__(self, name, session, foreground_color = (0,255,0,255), saturation = 10,
                 color_popup = False, flip_horizontal = True,
                 camera_name = None, size = None, framerate = 25):
        Model.__init__(self, name, session)

        self._camera = None		# QCamera
        self.camera_name = camera_name
        self._capture = None		# VideoCapture instance
        self._capture_started = False
        self.size = None		# Width, height in pixels
        self._requested_size = size
        self.framerate = None		# Frames per second, float
        self._requested_framerate = framerate
        self._first_image = True	# Whether first image has been acquired
        self._last_image = None		# Numpy color array (h,w,3) uint8
        self.foreground_color = foreground_color  # What color pixels to put in front of models
        self.saturation = saturation	# How saturated pixel color must be to be foreground
        self.color_popup = color_popup
        self._flip_horizontal = flip_horizontal

        self._start_video()

    def delete(self):
        self._close_camera()
        Model.delete(self)

    def _close_camera(self):
        cam = self._camera
        if cam:
            cam.stop()
            cam.deleteLater()
            self._camera = None

        cap = self._capture_session
        if cap:
            cap.setCamera(None)		# Avoid crash deleting capture session
            self._capture_session = None

        self._video_capture = None
        
    def _start_video(self):
        cam_device = self._find_camera()
        from Qt.QtMultimedia import QCamera
        cam = QCamera(cam_device)
        self._camera = cam
        cam.errorOccurred.connect(self._camera_error)

        format = self._choose_video_settings()
        cam.setCameraFormat(format)
        r = format.resolution()
        w,h = r.width(), r.height()
        rate = format.maxFrameRate()
        msg = ('Using web camera "%s", width %d, height %d, framerate %.4g'
               % (self.camera_name, w, h, rate))
        self.session.logger.info(msg)
        
        from Qt.QtMultimedia import QMediaCaptureSession, QVideoSink
        cap_ses = QMediaCaptureSession()
        self._capture_session = cap_ses
        cap_ses.setCamera(cam)
        cap_sink = VideoCapture(self._new_frame)
        self._video_capture = cap_sink
        cap_ses.setVideoSink(cap_sink)

        cam.start()

    def _find_camera(self):
        from Qt.QtMultimedia import QMediaDevices
        if self.camera_name is None:
            device = QMediaDevices.defaultVideoInput()
            self.camera_name = device.description()
        else:
            cam_list = QMediaDevices.videoInputs()
            if len(cam_list) == 0:
                from chimerax.core.errors import UserError
                raise UserError('Did not find any cameras')
            cam_matches = [device for device in cam_list
                           if device.description == self.camera_name]
            if len(cam_matches) == 0:
                from chimerax.core.errors import UserError
                raise UserError('Did not find camera named "%s"' % cam_name)
            if len(cam_matches) > 1:
                self.session.logger.info('Found multiple cameras: %s'
                                         % ', '.join(c.description() for c in cam_list))
            device = cam_matches[0]
        return device
        
    def _choose_video_settings(self):
        '''
        Choose camera pixel format, resolution and framerate.
        Return QViewfinderSettings instance.
        '''
        cam = self._camera
        cam_device = cam.cameraDevice()
        available = [format for format in cam_device.videoFormats()
                     if format.pixelFormat() in VideoCapture.supported_formats]
        if len(available) == 0:
            pformats = set(format.pixelFormat() for format in cam_device.videoFormats())
            cam_formats = ','.join(_qt_pixel_format_name(f) for f in pformats)
            sformats = VideoCapture.supported_formats
            sup_formats = ','.join(_qt_pixel_format_name(f) for f in sformats)
            from chimerax.core.errors import UserError
            raise UserError(f'Camera "{self.camera_name}"'
                            f' pixel formats ({cam_formats})'
                            f' are not supported by webcam command ({sup_formats})')

        # Find the best match for the requested resolution and framerate.
        size = self._requested_size
        fps = self._requested_framerate
        def match_score(format, size=size, fps=fps):
            r = format.resolution()
            w,h = r.width(), r.height()
            fr = format.maxFrameRate()
            if size is None and fps is None:
                return (w*h, fr)
            elif size is not None and fps is None:
                return (-(abs(w-size[0]) + abs(h-size[1])), fr)
            elif size is None and fps is not None:
                return (-abs(fr-fps), w*h)
            else:
                return (-(abs(w-size[0]) + abs(h-size[1])), -abs(fr-fps))
        format = max(available, key = match_score)
        return format

    def _camera_error(self, error, message):
        print ('camera error:', message)
        
    def _new_frame(self, rgba_image):
        '''rgba_image is a uint8 numpy array of shape (height, width, 4).'''
        if self._camera is None:
            # Get one new frame after camera unload is requested.
            return

        # Video frame can change size when some other app opens the camera
        # and changes its settings.  ChimeraX ticket #3517
        h,w = rgba_image.shape[:2]
        self.size = (w,h)
        
        self._mark_image_foreground(rgba_image)
        
        self._last_image = rgba_image
        
        if self._first_image:
            self._create_video_texture(rgba_image)
            self._first_image = False
        else:
            self.texture.reload_texture(rgba_image)
            self.redraw_needed()

    def _mark_image_foreground(self, rgba_image):
        '''
        Set image alpha values to 0 for pixels that should appear in front of models.
        '''
        fg = self.foreground_color
        sat = int(2.55*self.saturation)  # Convert 0-100 scale to 0-255
        from .webcam_cpp import set_color_alpha
        set_color_alpha(rgba_image, fg, sat, 0)
        return

        # Unused Python version
        im = rgba_image
        rgb = [im[:,:,c] for c in (0,1,2)]
        fg_mask = None
        from numpy import logical_and
        for j,k in ((0,1), (1,2), (0,2)):
            if fg[j] != fg[k]:
                l,s = (j,k) if fg[j]>fg[k] else (k,j)
                mask = (rgb[l] >= sat)	# Avoid wrap of uint8 values.
                if fg_mask is None:
                    fg_mask = mask
                else:
                    logical_and(fg_mask, mask, fg_mask)	# Avoid wrap of uint8 values.
                logical_and(fg_mask, (rgb[l]-sat > rgb[s]), fg_mask)
        alpha = im[:,:,3]
        alpha[fg_mask] = 0

    def _create_video_texture(self, color_image):
        # TODO: Does not have sensible bounds.  Bounds don't really make sense.
        #       Causes surprises if it is the first model opened.
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, color_image, (-1, -1), (2, 2))
        self.vertices[:,2] = 0.99 	# Put at maximum depth.
        # Invert y-axis by flipping texture coordinates
        tc = self.texture_coordinates
        tc[:,1] = 1 - tc[:,1]
        if self._flip_horizontal:
            # This makes it easier to point at a model while looking
            # at yourself on the screen
            tc[:,0] = 1 - tc[:,0]
        self.texture_coordinates = tc

    def _get_flip_horizontal(self):
        return self._flip_horizontal
    def _set_flip_horizontal(self, flip):
        if flip != self._flip_horizontal:
            self._flip_horizontal = flip
            tc = self.texture_coordinates
            tc[:,0] = 1 - tc[:,0]
            self.texture_coordinates = tc
    flip_horizontal = property(_get_flip_horizontal, _set_flip_horizontal)
    
    def draw(self, renderer, draw_pass):
        '''Render a color and depth texture pair.'''
        if self._first_image:
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
        r.enable_capabilities |= r.SHADER_ALPHA_DEPTH

        # If the desired field of view of the texture does not match the camera field of view
        # then adjust projection size.  Also if the apect ratio of the target framebuffer and
        # the aspect ratio of the texture don't match adjust the projection size.
        w,h = r.render_size()
        vw,vh = self.size
        if vw*h > vh*w:
            # Video aspect is wider than window aspect. Fit height, clip width.
            sx = (vw/vh) / (w/h)
            sy = 1
        else:
            # Video aspect is narrower than window aspect. Fit width, clip height.
            sx = 1
            sy = (w/h) / (vw/vh)
        
        cur_proj = r.current_projection_matrix
        proj = ((sx, 0, 0, 0), (0, sy, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        r.set_projection_matrix(proj)

        from chimerax.geometry import place
        p0 = place.identity()
        cur_view = r.current_view_matrix
        r.set_view_matrix(p0)
        r.set_model_matrix(p0)
        r.enable_blending(False)

        Model.draw(self, r, draw_pass)
            
        # Restore view and projection matrices since drawings are not supposed to change these.
        r.set_projection_matrix(cur_proj)
        r.set_view_matrix(cur_view)
        
        r.enable_capabilities &= ~r.SHADER_ALPHA_DEPTH
        r.disable_shader_capabilities(0)

    def set_window_size(self):
        w,h = self.size
        from chimerax.core.commands import run
        run(self.session, 'windowsize %d %d' % (w,h))
  
    # ---------------------------------------------------------------------------
    #
    def first_intercept(self, mxyz1, mxyz2, exclude = None):
        '''
        Pick on video and provide the color as the description, so mouse hover
        can report color under mouse.
        This is a hack.  The segment mxyz1, mxyz2 parameters are ignored and
        instead the mouse position is used because it is hard to convert from
        scene coordinates to window coordinates.
        '''
        if not self.color_popup:
            return None
        
        if exclude is not None and exclude(self):
            return None

        gw = self.session.ui.main_window.graphics_window
        w,h = gw.width(), gw.height()
        if w == 0 or h == 0:
            return None
        
        im = self._last_image
        if im is None:
            return None

        from Qt.QtGui import QCursor
        p = gw.mapFromGlobal(QCursor.pos())
        x,y = p.x(), p.y()
        ih, iw = im.shape[:2]
        siw, sih = (iw, h*(iw/w)) if w*ih > h*iw else (w*(ih/h), ih)  # Visible part of image
        ix, iy = int((iw-siw)/2 + (x/w)*siw), int((ih-sih)/2 + (y/h)*sih)
        if self._flip_horizontal:
            ix = (iw-1) - ix
        if ix < 0 or ix >= iw or iy < 0 or iy >= ih:
            return None
        
        color = tuple(int(100*r/255) for r in im[iy,ix,:3])
#        print ('graphics window cursor position', x, y, w, h, 'image pos', ix, iy, siw, sih, 'color', color)
        pick = ColorPick(color)
        
        return pick
    
# -----------------------------------------------------------------------------
#
from chimerax.graphics import Pick
class ColorPick(Pick):
    def __init__(self, rgb, depth=1):
        Pick.__init__(self, depth)
        self.rgb = rgb
    def description(self):
        return 'Color %d,%d,%d' % self.rgb

# -----------------------------------------------------------------------------
#
from Qt.QtMultimedia import QVideoSink, QVideoFrameFormat
class VideoCapture(QVideoSink):
    
    supported_formats = (QVideoFrameFormat.PixelFormat.Format_ARGB8888,
                         QVideoFrameFormat.PixelFormat.Format_YUYV,
                         QVideoFrameFormat.PixelFormat.Format_UYVY)

    bytes_per_pixel = {QVideoFrameFormat.PixelFormat.Format_ARGB8888:4,
                       QVideoFrameFormat.PixelFormat.Format_YUYV:2,
                       QVideoFrameFormat.PixelFormat.Format_UYVY:2}

    def __init__(self, new_frame_cb):
        self._rgba_image = None		# Numpy uint8 array of size (h, w, 4)
        self._new_frame_cb = new_frame_cb
        QVideoSink.__init__(self)
        self.videoFrameChanged.connect(self._got_frame)

    def _got_frame(self):
        frame = self.videoFrame()
        self._rgba_image = _numpy_rgba_array_from_qt_video_frame(frame, self._rgba_image)
        self._new_frame_cb(self._rgba_image)

# -----------------------------------------------------------------------------
#
_wrong_frame_size = False
def _numpy_rgba_array_from_qt_video_frame(frame, rgba_image = None):
    f = frame
    pixel_format = f.pixelFormat()
    if pixel_format not in VideoCapture.supported_formats:
        raise ValueError('Cannot convert QVideoFrame with pixel format %d to numpy array'
                         % pixel_format)

    # Map video frame into memory.
    from Qt.QtMultimedia import QVideoFrame
    f.map(QVideoFrame.MapMode.ReadOnly)

    # Check video frame size
    w, h = f.width(), f.height()
    if rgba_image is not None:
        ih, iw = rgba_image.shape[:2]
        if ih != h or iw != w:
            # Video frame does not match array size so make new array.
            rgba_image = None
    bytes_per_pixel = VideoCapture.bytes_per_pixel[pixel_format]
    nbytes = h * w * bytes_per_pixel
    plane = 0
    if f.mappedBytes(plane) != nbytes:
        global _wrong_frame_size
        if not _wrong_frame_size:
            # On 2012 MacBookPro, 1280x720 gives 3686432 mapped bytes instead of 3686400.
            # Dropping last 32 bytes gives correct image.
            _wrong_frame_size = True
            print ('QVideoFrame (%d by %d, pixel format %d) has wrong number of bytes %d, expected %d' 
                   % (f.width(), f.height(), f.pixelFormat(), f.mappedBytes(plane), nbytes))

    # Create rgba image array
    if rgba_image is None:
        from numpy import empty, uint8
        rgba_image = empty((h,w,4), uint8)

    # Convert video frame data to rgba
    if f.planeCount() != 1:
        raise ValueError('Expected video plane count to be 1, got %d' % pixel_format)
    data = f.bits(plane)	# sip.voidptr
    pointer = int(data)		# Convert sip.voidptr to integer to pass to C++ code.
    from .webcam_cpp import bgra_to_rgba, yuyv_to_rgba, uyvy_to_rgba
    from Qt.QtMultimedia import QVideoFrameFormat
    if pixel_format == QVideoFrameFormat.PixelFormat.Format_ARGB8888:
        bgra_to_rgba(pointer, rgba_image)
#        _bgra_to_rgba(data, rgba_image)
    elif pixel_format == QVideoFrameFormat.PixelFormat.Format_YUYV:
        yuyv_to_rgba(pointer, rgba_image)
#        _yuyv_to_rgba(data, rgba_image)
    elif pixel_format == QVideoFrameFormat.PixelFormat.Format_UYVY:
        uyvy_to_rgba(pointer, rgba_image)

    # Release mapped video frame data.
    f.unmap()

    return rgba_image

# -----------------------------------------------------------------------------
#
def _bgra_to_rgba(data, rgba_image):
    # TODO: add an array argument and avoid making temporary arrays.
    #  Best done in C++.
    h,w = rgba_image.shape[:2]
    buf = data.asstring(h*w*4)
    from numpy import uint8, frombuffer
    bgra = frombuffer(buf, uint8).reshape((h, w, 4))
    
    rgba_image[:,:,0] = bgra[:,:,2]
    rgba_image[:,:,1] = bgra[:,:,1]
    rgba_image[:,:,2] = bgra[:,:,0]
    rgba_image[:,:,3] = bgra[:,:,3]

# -----------------------------------------------------------------------------
#
def _yuyv_to_rgba(data, rgba_image):
    h,w = rgba_image.shape[:2]
    buf = data.asstring(h*w*2)
    from numpy import uint8, frombuffer
    yuyv = frombuffer(buf, uint8).reshape((h, w, 2))
    _yuyv_to_rgba(yuyv, rgba_image)

# -----------------------------------------------------------------------------
#
def _yuyv_to_rgba(yuyv, rgba_image):
    yuv = _yuyv_to_yuv(yuyv)
    _yuv_to_rgba(yuv, rgba_image)

# -----------------------------------------------------------------------------
#
def _yuyv_to_yuv(yuyv):
    h,w = yuyv.shape[:2]
    from numpy import empty, uint8
    yuv = empty((h,w,3), uint8)
    y = yuyv[:,:,0]
    u = yuyv[:,::2,1]
    v = yuyv[:,1::2,1]
    yuv[:,:,0] = y
    yuv[:,::2,1] = u
    yuv[:,1::2,1] = u
    yuv[:,::2,2] = v
    yuv[:,1::2,2] = v
    return yuv

# -----------------------------------------------------------------------------
#
def _yuv_to_rgba(yuv, rgba_image):
# From https://www.fourcc.org/fccyvrgb.php
# B = 1.164(Y - 16)                   + 2.018(U - 128)
# G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
# R = 1.164(Y - 16) + 1.596(V - 128)
    from numpy import float32, clip
    y, u, v = yuv[:,:,0].astype(float32), yuv[:,:,1].astype(float32), yuv[:,:,2].astype(float32)
    b = 1.164 * (y - 16)                   + 2.018 * (u - 128)
    g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)
    r = 1.164 * (y - 16) + 1.596 * (v - 128)
    h,w = yuv.shape[:2]
    clip(r, 0, 255, rgba_image[:,:,0])
    clip(g, 0, 255, rgba_image[:,:,1])
    clip(b, 0, 255, rgba_image[:,:,2])
    rgba_image[:,:,3] = 255

# -----------------------------------------------------------------------------
#
_qt_pixel_format_names = {}
def _qt_pixel_format_name(qt_pixel_format):
    global _qt_pixel_format_names
    if len(_qt_pixel_format_names) == 0:
        from Qt.QtMultimedia import QVideoFrameFormat
        for name in dir(QVideoFrameFormat):
            if name.startswith('Format_'):
                value = getattr(QVideoFrameFormat, name)
                _qt_pixel_format_names[value] = name[7:]
    name = _qt_pixel_format_names.get(qt_pixel_format)
    if name is None:
        name = str(qt_pixel_format)
    return name
