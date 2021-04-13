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
def webcam(session, enable = True, foreground_color = (0,255,0,255), saturation = 5,
           flip_horizontal = True, name = None, size = None, framerate = 25, color_popup = False):

    wc_list = session.models.list(type = WebCam)
    if enable:
        if len(wc_list) == 0:
            wc = WebCam('webcam', session,
                        foreground_color = foreground_color, saturation = saturation,
                        flip_horizontal = flip_horizontal, color_popup = color_popup,
                        camera_name = name, size = size, framerate = framerate)
            session.models.add([wc])
        else:
            wc = wc_list[0]
            wc.foreground_color = foreground_color
            wc.saturation = saturation
            wc.flip_horizontal = flip_horizontal
            wc.color_popup = color_popup
        
#        if set_window_size:
#            wc.set_window_size()
    else:
        session.models.close(wc_list)
            
# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import BoolArg, Color8Arg, IntArg, FloatArg, Int2Arg, StringArg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('foreground_color', Color8Arg),
                              ('saturation', IntArg),
                              ('flip_horizontal', BoolArg),
                              ('color_popup', BoolArg),
                              ('name', StringArg),
                              ('size', Int2Arg),
                              ('framerate', FloatArg)],
                   synopsis = 'Turn on webcam rendering')
    register('webcam', desc, webcam, logger=logger)
    create_alias('device webcam', 'webcam $*', logger=logger)
            
# -----------------------------------------------------------------------------
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
            cam.unload()
            cam.deleteLater()
            self._camera = None
        
    def _start_video(self):
        cam_info = self._find_camera()
        from Qt.QtMultimedia import QCamera
        cam = QCamera(cam_info)
        self._camera = cam
#        print('camera availability (0 = available):', cam.availability())
        cam.setCaptureMode(QCamera.CaptureVideo)
        cam.statusChanged.connect(self._camera_status_changed)
        import sys
        if sys.platform == 'linux':
            # Ubuntu 18.04 with gstreamer and Qt 5.15.1 needs view finder
            # to be set before starting camera.  Ticket #3976
            capture = VideoCapture(self._new_frame)
            self._capture = capture
            cam.setViewfinder(capture)
        cam.start()

    def _find_camera(self):
        from Qt.QtMultimedia import QCameraInfo
        cam_list = QCameraInfo.availableCameras()
        if len(cam_list) == 0:
            from chimerax.core.errors import UserError
            raise UserError('Did not find any cameras')
        if len(cam_list) > 1:
            self.session.logger.info('Found multiple cameras: %s'
                                     % ', '.join(c.description() for c in cam_list))
        cam_name = self.camera_name
        if cam_name is None:
            cam_info = cam_list[0]
            self.camera_name = cam_info.description()
        else:
            cam_info = None
            for cinfo in cam_list:
                if cinfo.description() == cam_name:
                    cam_info = cinfo
                    break
            if cam_info is None:
                from chimerax.core.errors import UserError
                raise UserError('Did not find camera named "%s"' % cam_name)
        #self.session.logger.info('Using camera "%s"' % cam_info.description())
        return cam_info

    def _camera_status_changed(self, status):
#        print ('current camera status', status)

        from Qt.QtMultimedia import QCamera
        if status == QCamera.ActiveStatus and not self._capture_started:
            self._start_capture()

    def _start_capture(self):
        
        cam = self._camera
        import sys
        if sys.platform != 'linux':
            capture = VideoCapture(self._new_frame)
            self._capture = capture
            cam.setViewfinder(capture)
            print ('set view finder')
        # self._report_supported_pixel_formats()
        # Have to set camera pixel format after setting camera view finder
        # otherwise it does not seem to know the available pixel formats
        # and just uses default pixel format.
        try:
            settings = self._choose_camera_settings()
        except Exception:
            # Pixel format not offered by camera.
            self._close_camera()
            raise

        self._capture_started = True

        # Need to stop the camera on macOS 10.15 laptop or a new
        # resolution is ignored.
        cam.stop()
        cam.setViewfinderSettings(settings)
        cam.start()

        # Record actual settings.
        settings = cam.viewfinderSettings()
        res = settings.resolution()
        w, h = res.width(), res.height()
        self.size = (w,h)
        self.framerate = settings.maximumFrameRate()

        msg = ('Using web camera "%s", width %d, height %d, framerate %.4g'
                       % (self.camera_name, w, h, self.framerate))
        self.session.logger.info(msg)

        # Start acquiring images.
        from Qt.QtMultimedia import QVideoSurfaceFormat
        from Qt.QtCore import QSize
        fmt = QVideoSurfaceFormat(QSize(w,h), settings.pixelFormat())
        self._capture.start(fmt)
        
    def _choose_camera_settings(self):
        '''
        Choose camera pixel format, resolution and framerate.
        Return QViewfinderSettings instance.
        '''
        cam = self._camera
        available = [settings for settings in cam.supportedViewfinderSettings()
                     if settings.pixelFormat() in VideoCapture.supported_formats]
        if len(available) == 0:
            pformats = cam.supportedViewfinderPixelFormats()
            from chimerax.core.errors import UserError
            raise UserError('Camera "%s" pixel formats (%s) are not supported by webcam command (%s)'
                            % (self.camera_name,
                               ','.join(_qt_pixel_format_name(f) for f in pformats),
                               ','.join(_qt_pixel_format_name(f) for f in VideoCapture.supported_formats)))

        # Find the best match for the requested resolution and framerate.
        size = self._requested_size
        fps = self._requested_framerate
        def match_score(settings, size=size, fps=fps):
            r = settings.resolution()
            w,h = r.width(), r.height()
            fr = settings.maximumFrameRate()
            if size is None and fps is None:
                return (w*h, fr)
            elif size is not None and fps is None:
                return (-(abs(w-size[0]) + abs(h-size[1])), fr)
            elif size is None and fps is not None:
                return (-abs(fr-fps), w*h)
            else:
                return (-(abs(w-size[0]) + abs(h-size[1])), -abs(fr-fps))
        settings = s = max(available, key = match_score)
        return settings
        
    def _report_supported_pixel_formats(self):
        cam = self._camera
        pformats = cam.supportedViewfinderPixelFormats()
        print ('supported pixel formats', pformats)
        ss = cam.supportedViewfinderSettings()
        print('supported view finder settings', len(ss))
        for i,s in enumerate(ss):
            sz = s.resolution()
            pf = s.pixelFormat()
            frmin, frmax = s.minimumFrameRate(), s.maximumFrameRate()
            print ('%d: res %d %d, format %d, framerate %g - %g'
                   % (i, sz.width(), sz.height(), pf, frmin, frmax))

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
            print ('set flip horizontal', flip)
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
from Qt.QtMultimedia import QAbstractVideoSurface, QVideoFrame
class VideoCapture(QAbstractVideoSurface):
    
    supported_formats = (QVideoFrame.Format_ARGB32, QVideoFrame.Format_YUYV)
#    supported_formats = (QVideoFrame.Format_YUYV, QVideoFrame.Format_ARGB32)

    def __init__(self, new_frame_cb):
        self._rgba_image = None		# Numpy uint8 array of size (h, w, 4)
        self._new_frame_cb = new_frame_cb
        QAbstractVideoSurface.__init__(self)

    def present(self, frame):
        if frame.isValid():
            self._rgba_image = _numpy_rgba_array_from_qt_video_frame(frame, self._rgba_image)
            self._new_frame_cb(self._rgba_image)
        return True

    def isFormatSupported(self, format):
        return format in self.supported_formats

    def start(self, format):
        return QAbstractVideoSurface.start(self, format)
    
    def stop(self):
        QAbstractVideoSurface.stop(self)
        
    def supportedPixelFormats(self, handle_type):
        return self.supported_formats

# -----------------------------------------------------------------------------
#
_wrong_frame_size = False
def _numpy_rgba_array_from_qt_video_frame(frame, rgba_image = None):
    f = frame
    pixel_format = f.pixelFormat()
    if pixel_format not in (f.Format_ARGB32, f.Format_YUYV):
        raise ValueError('Cannot convert QVideoFrame with pixel format %d to numpy array' % pixel_format)

    # Map video frame into memory.
    from Qt.QtMultimedia import QAbstractVideoBuffer
    f.map(QAbstractVideoBuffer.ReadOnly)

    # Check video frame size
    w, h = f.width(), f.height()
    if rgba_image is not None:
        ih, iw = rgba_image.shape[:2]
        if ih != h or iw != w:
            # Video frame does not match array size so make new array.
            rgba_image = None
    bytes_per_pixel = {f.Format_ARGB32:4, f.Format_YUYV:2}[pixel_format]
    nbytes = h * w * bytes_per_pixel
    if f.mappedBytes() != nbytes:
        global _wrong_frame_size
        if not _wrong_frame_size:
            # On 2012 MacBookPro, 1280x720 gives 3686432 mapped bytes instead of 3686400.
            # Dropping last 32 bytes gives correct image.
            _wrong_frame_size = True
            print ('QVideoFrame (%d by %d, pixel format %d) has wrong number of bytes %d, expected %d' 
                   % (f.width(), f.height(), f.pixelFormat(), f.mappedBytes(), nbytes))

    # Create rgba image array
    if rgba_image is None:
        from numpy import empty, uint8
        rgba_image = empty((h,w,4), uint8)

    # Convert video frame data to rgba
    data = f.bits()		# sip.voidptr
    pointer = int(data)		# Convert sip.voidptr to integer to pass to C++ code.
    from .webcam_cpp import bgra_to_rgba, yuyv_to_rgba
    if pixel_format == f.Format_ARGB32:
        a = bgra_to_rgba(pointer, rgba_image)
#        a = _bgra_to_rgba(data, rgba_image)
    elif pixel_format == f.Format_YUYV:
        a = yuyv_to_rgba(pointer, rgba_image)
#        a = _yuyv_to_rgba(data, rgba_image)

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
        from Qt.QtMultimedia import QVideoFrame    
        for name in dir(QVideoFrame):
            if name.startswith('Format_'):
                value = getattr(QVideoFrame, name)
                _qt_pixel_format_names[value] = name[7:]
    name = _qt_pixel_format_names.get(qt_pixel_format)
    if name is None:
        name = str(int(qt_pixel_format))
    return name
