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
def webcam(session, enable = True, foreground_color = (0,255,0,255), saturation = 10,
           flip_horizontal = True, color_popup = False):

    wc_list = session.models.list(type = WebCam)
    if enable:
        if len(wc_list) == 0:
            wc = WebCam('webcam', session,
                        foreground_color = foreground_color, saturation = saturation,
                        flip_horizontal = flip_horizontal, color_popup = color_popup)
            session.models.add([wc])
            w,h = wc.size
            msg = ('Web camera "%s", width %d, height %d, framerate %.4g'
                   % (wc.camera_name, w, h, wc.framerate))
            session.logger.info(msg)
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
    from chimerax.core.commands import CmdDesc, register, create_alias, BoolArg, Color8Arg, IntArg
    desc = CmdDesc(optional = [('enable', BoolArg)],
                   keyword = [('foreground_color', Color8Arg),
                              ('saturation', IntArg),
                              ('flip_horizontal', BoolArg),
                              ('color_popup', BoolArg)],
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
                 color_popup = False, flip_horizontal = True):
        Model.__init__(self, name, session)

        self._camera = None		# QCamera
        self.camera_name = ''
        self._capture = None		# VideoCapture instance
        self.size = None		# Width, height in pixels
        self.framerate = None		# Frames per second, float
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
        from PyQt5.QtMultimedia import QCameraInfo, QCamera, QVideoFrame
        cam_list = QCameraInfo.availableCameras()
        if len(cam_list) == 0:
            from chimerax.core.errors import UserError
            raise UserError('Did not find any cameras')
        if len(cam_list) > 1:
            self.session.logger.info('Found multiple cameras: %s'
                                     % ', '.join(c.description() for c in cam_list))
        cam_info = cam_list[0]
        self.camera_name = cam_info.description()
#        self.session.logger.info('Using camera "%s"' % cam_info.description())

        cam = QCamera(cam_info)
        self._camera = cam
#        print('camera availability (0 = available):', cam.availability())
        cam.setCaptureMode(QCamera.CaptureVideo)
        cam.stateChanged.connect(self._camera_state_changed)
        cam.start()

#        self._start_capture()

    def _camera_state_changed(self, state):
#        print ('current camera state', state)

        from PyQt5.QtMultimedia import QCamera
        if state == QCamera.ActiveState and self._capture is None:
            self._start_capture()

#        cam = self._camera
#        print ('capture mode (2=video)', int(cam.captureMode()))
#        res = cam.supportedViewfinderResolutions()
#        print ('supported resolutions', [(s.width(), s.height()) for s in res])
#        frates = cam.supportedViewfinderFrameRateRanges()
#        print ('supported framerate ranges', [(fr.minimumFrameRate, fr.maximumFrameRate) for fr in frates])
#        pformats = cam.supportedViewfinderPixelFormats()
#        print ('supported pixel formats', pformats)
#        self._start_capture()

    def _start_capture(self):
        
        cam = self._camera
        settings = cam.viewfinderSettings()

        res = settings.resolution()
        w, h = res.width(), res.height()
        self.size = (w,h)
        frmin, frmax = settings.minimumFrameRate(), settings.maximumFrameRate()
        self.framerate = frmax

        capture = VideoCapture(self._new_frame)
        self._capture = capture
        cam.setViewfinder(capture)
        # Have to set camera pixel format after setting camera view finder
        # otherwise it does not seem to know the available pixel formats
        # and just uses default pixel format.
        from PyQt5.QtMultimedia import QVideoSurfaceFormat, QVideoFrame
        try:
            pixel_format = self._set_camera_pixel_format()
        except Exception:
            # Pixel format not offered by camera.
            self._close_camera()
            raise
        w,h = self.size
        from PyQt5.QtCore import QSize
        fmt = QVideoSurfaceFormat(QSize(w,h), pixel_format)
        capture.start(fmt)

    def _set_camera_pixel_format(self):
        cam = self._camera
        settings = cam.viewfinderSettings()
        pixel_format = settings.pixelFormat()
        if pixel_format not in VideoCapture.supported_formats:
            pformats = set(cam.supportedViewfinderPixelFormats())
            for pixel_format in VideoCapture.supported_formats:
                if pixel_format in pformats:
                    break
            else:
                from chimerax.core.errors import UserError
                raise UserError('Camera "%s" pixel formats (%s) are not supported by webcam command (%s)'
                                % (self.camera_name,
                                   ','.join(_qt_pixel_format_name(f) for f in pformats),
                                   ','.join(_qt_pixel_format_name(f) for f in VideoCapture.supported_formats)))
            from PyQt5.QtMultimedia import QCameraViewfinderSettings
            new_settings = QCameraViewfinderSettings(settings)
            new_settings.setPixelFormat(pixel_format)
            cam.setViewfinderSettings(new_settings)
        return pixel_format
    
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

        self._mark_image_foreground(rgba_image)
        
        self._last_image = rgba_image
            
        if self._first_image:
            self._create_video_texture(rgba_image)
            self._first_image = False
            #print('first frame shape', a.shape, 'type', a.dtype, 'pixel format', f.pixelFormat())
        else:
            self.texture.reload_texture(rgba_image)
            self.redraw_needed()

    def _mark_image_foreground(self, rgba_image):
        '''
        Set image alpha values to 0 for pixels that should appear in front of models.
        '''
        # TODO: This should be done in C++ for speed
        im = rgba_image
        rgb = [im[:,:,c] for c in (0,1,2)]
        fg = self.foreground_color
        sat = self.saturation
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

        from PyQt5.QtGui import QCursor
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
from PyQt5.QtMultimedia import QAbstractVideoSurface, QVideoFrame
class VideoCapture(QAbstractVideoSurface):
    
#    supported_formats = (QVideoFrame.Format_ARGB32, QVideoFrame.Format_YUYV)
    supported_formats = (QVideoFrame.Format_YUYV, QVideoFrame.Format_ARGB32)

    def __init__(self, new_frame_cb):
        self._new_frame_cb = new_frame_cb
        QAbstractVideoSurface.__init__(self)

    def present(self, frame):
        if frame.isValid():
            rgba_image = _numpy_rgba_array_from_qt_video_frame(frame)
            self._new_frame_cb(rgba_image)
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
def _numpy_rgba_array_from_qt_video_frame(frame):
    f = frame
    pixel_format = f.pixelFormat()
    if pixel_format == f.Format_ARGB32:
        a = _qvideoframe_argb32_to_numpy(f)
    elif pixel_format == f.Format_YUYV:
        a = _qvideoframe_yuyv_to_numpy(f)
    else:
        raise ValueError('Cannot convert QVideoFrame with pixel format %d to numpy array' % pixel_format)
    return a

# -----------------------------------------------------------------------------
#
_wrong_frame_size = False
def _qvideoframe_argb32_to_numpy(frame):
    # TODO: add an array argument and avoid making temporary arrays.
    #  Best done in C++.
    f = frame
    from PyQt5.QtMultimedia import QAbstractVideoBuffer
    f.map(QAbstractVideoBuffer.ReadOnly)
    shape = (f.height(), f.width(), 4)
    nbytes = f.height() * f.width() * 4
    global _wrong_frame_size
    if not _wrong_frame_size and f.mappedBytes() != nbytes:
        # On 2012 MacBookPro, 1280x720 gives 3686432 mapped bytes instead of 3686400.
        # Dropping last 32 bytes gives correct image.
        _wrong_frame_size = True
        print ('QVideoFrame (%d by %d, pixel format %d) has wrong number of bytes %d, expected %d'
               % (f.width(), f.height(), f.pixelFormat(), f.mappedBytes(), nbytes))
    buf = f.bits().asstring(nbytes)
    from numpy import uint8, frombuffer
    bgra = frombuffer(buf, uint8).reshape(shape)
    f.unmap()
    rgba = bgra.copy()
    rgba[:,:,0] = bgra[:,:,2]
    rgba[:,:,2] = bgra[:,:,0]
    return rgba

# -----------------------------------------------------------------------------
#
def _qvideoframe_yuyv_to_numpy(frame):
    f = frame
    w,h = f.width(), f.height()
    from PyQt5.QtMultimedia import QAbstractVideoBuffer
    f.map(QAbstractVideoBuffer.ReadOnly)
    buf = f.bits().asstring(f.mappedBytes())
    f.unmap()
    from numpy import uint8, frombuffer
    yuyv = frombuffer(buf, uint8).reshape((h, w, 2))
    yuv = _yuyv_to_yuv(yuyv)
    rgba = _yuv_to_rgba(yuv)
    return rgba

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
def _yuv_to_rgba(yuv):
# From https://www.fourcc.org/fccyvrgb.php
# B = 1.164(Y - 16)                   + 2.018(U - 128)
# G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
# R = 1.164(Y - 16) + 1.596(V - 128)
    from numpy import float32, empty, uint8, clip
    y, u, v = yuv[:,:,0].astype(float32), yuv[:,:,1].astype(float32), yuv[:,:,2].astype(float32)
    b = 1.164 * (y - 16)                   + 2.018 * (u - 128)
    g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)
    r = 1.164 * (y - 16) + 1.596 * (v - 128)
    h,w = yuv.shape[:2]
    rgba = empty((h,w,4), uint8)
    clip(r, 0, 255, rgba[:,:,0])
    clip(g, 0, 255, rgba[:,:,1])
    clip(b, 0, 255, rgba[:,:,2])
    rgba[:,:,3] = 255
    return rgba

# -----------------------------------------------------------------------------
#
_qt_pixel_format_names = {}
def _qt_pixel_format_name(qt_pixel_format):
    global _qt_pixel_format_names
    if len(_qt_pixel_format_names) == 0:
        from PyQt5.QtMultimedia import QVideoFrame    
        for name in dir(QVideoFrame):
            if name.startswith('Format_'):
                value = getattr(QVideoFrame, name)
                _qt_pixel_format_names[value] = name[7:]
    name = _qt_pixel_format_names.get(qt_pixel_format)
    if name is None:
        name = str(int(qt_pixel_format))
    return name
