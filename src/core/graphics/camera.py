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

'''
Camera
======
'''

class Camera:
    '''
    A Camera has a position in the scene and viewing direction given
    by a Place object.  The -z axis of the coordinate frame given by
    the Place object is the view direction.  The x and y axes are the
    horizontal and vertical axes of the camera frame.  Different cameras
    handle perspective projection, orthographic projection, 360 degree
    equirectangular projection, sequential stereo with shutter glasses,
    ....
    '''
    def __init__(self):

        # Camera postion and direction, neg z-axis is camera view direction,
        # x and y axes are horizontal and vertical screen axes.
        # First 3 columns are x, y, z axes, 4th column is camara location.
        from ..geometry import Place
        self._position = Place()
        """Coordinate frame for camera in scene coordinates with camera
        pointed along -z."""

        self._pixel_shift = 0, 0
        '''
        Shift the camera by this number of pixels in x and y from the
        geometric center of the window.  This is used for supersampling
        where fractional pixel shifts are used and the resulting images
        are blended.
        '''

        self.redraw_needed = False
        """Indicates whether a camera change has been made which requires
        the graphics to be redrawn."""

    name = 'unknown'
    '''Name indicating the type of camera, for example, "mono", "stereo", "orthographic".'''

    def get_position(self, view_num=None):
        p = self._position if view_num is None else self.view(self._position, view_num)
        return p

    def set_position(self, p):
        self._position = p
        self.redraw_needed = True
    position = property(get_position, set_position)
    '''Place object giving location and orientation of the camera in scene coordinates.
    Camera points along -z axis.'''

    def view_direction(self, view_num=None):
        '''The view direction of the camera in scene coordinates.'''
        return -self.get_position(view_num).z_axis()

    def number_of_views(self):
        '''
        TODO: Rename views to something clearer like "axis".
        Number of view points for this camera.  For example, sequential stereo has 2
        views for left and right eyes, and 360 stereo renders the 6 faces of a cube and
        has 6 views.'''
        return 1

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame for a specific camera view number.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        return camera_position

    def view_pixel_shift(self, view_num):
        '''
        Per view pixel shift of center away from center of render target.
        This is used for example to shift left/right eye images in sequential
        stereo camera.
        '''
        return (0,0)

    def set_fixed_pixel_shift(self, shift):
        '''
        Set per view pixel shift of center away from center of render target.
        Used for supersampled image capture.
        '''
        self._pixel_shift = shift

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  If the window size is given (width, height)
        in pixels then the models are fit in both height and width,
        otherwise just the width is fit. If pad is specified the fit
        is to a window size reduced by this fraction.
        The camera view direction is not changed.
        '''
        raise NotImplementedError('Camera has no view_all() method')

    def view_width(self, point):
        '''Return the full width of the view at the distance of point.
        The point is in scene coordinates.'''
        raise NotImplementedError('Camera has no view_width() method')

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        xps, yps = self._pixel_shift		# supersampling shift
        vxs, vys = self.view_pixel_shift(view_num)	# Per-view shift
        pixel_shift = (xps + vxs, yps + vys)
        return perspective_projection_matrix(self.field_of_view, window_size,
                                             near_far_clip, pixel_shift)

    def ray(self, window_x, window_y, window_size):
        '''
        Return origin and direction in scene coordinates of sight line
        for the specified window pixel position.
        '''
        return (None, None)

    def rectangle_bounding_planes(self, corner1, corner2, window_size):
        '''
        Planes as 4-vectors bounding the view through a window rectangle.
        Rectangle diagonally opposite corners are given by corner1 and corner2
        in pixels, and window size is in pixels.
        '''
        x1, y1 = corner1
        x2, y2 = corner2
        corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
        clockwise = ((x2 > x1) != (y1 > y2))
        if not clockwise:
            corners.reverse()
        rays = []
        for x,y in corners:
            origin, direction = self.ray(x, y, window_size)	# Scene coords
            if origin is None:
                return [] # Camera does not support ray calculation
            rays.append((origin,direction))
        (o1,d1),(o2,d2),(o3,d3),(o4,d4) = rays
        faces = ((o1,o1+d1,o2+d2), (o2,o2+d2,o3+d3), (o3,o3+d3,o4+d4), (o4,o4+d4,o1+d1))
        from .. import geometry
        planes = geometry.planes_as_4_vectors(faces)
        return planes

    def set_special_render_modes(self, render):
        '''
        Set any special rendering options needed by this camera.
        Called when this camera becomes the active camera.
        '''
        pass

    def clear_special_render_modes(self, render):
        '''
        Clear any special rendering options needed by this camera.
        Done when another camera becomes the active camera.
        '''
        pass

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        render.set_mono_buffer()

    def combine_rendered_camera_views(self, render):
        '''Combine camera views into a single image.'''
        pass

    def do_swap_buffers(self):
        return True

class MonoCamera(Camera):
    '''Perspective projection camera has an angular field of view measured in degrees.'''

    name = 'mono'
    
    def __init__(self):
        Camera.__init__(self)
        self.field_of_view = 30
        "Horizontal field of view in degrees."

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  The camera view direction is not changed.
        '''
        self.position = perspective_view_all(bounds, self.position, self.field_of_view, window_size, pad)

    def view_width(self, center):
        '''Return the width of the view at position center which is in
        scene coordinates.'''
        return perspective_view_width(center, self.position.origin(), self.field_of_view)

    def ray(self, window_x, window_y, window_size):
        '''
        Return origin and direction in scene coordinates of sight line
        for the specified window pixel position.
        '''
        d = perspective_direction(window_x, window_y, window_size, self.field_of_view)
        p = self.position
        ds = p.apply_without_translation(d)  # Convert camera to scene coordinates
        return (p.origin(), ds)

def perspective_view_all(bounds, position, field_of_view, window_size = None, pad = 0):
    '''
    Return the camera position that shows the specified bounds.
    Camera has perspective projection.
    '''
    from math import radians, sin, cos
    fov2 = radians(field_of_view)/2
    s,c = sin(fov2), cos(fov2)
    face_normals = [position.apply_without_translation(v)
                    for v in ((c/s,0,1), (-c/s,0,1))] # frustum side normals
    if window_size is not None:
        aspect = window_size[1]/window_size[0]
        from math import tan, atan
        fov2y = atan(aspect*tan(fov2))
        sy,cy = sin(fov2y), cos(fov2y)
        face_normals.extend([position.apply_without_translation(v)
                             for v in ((0,cy/sy,1), (0,-cy/sy,1))]) # frustum top/bottom normals
    center = bounds.center()
    bc = bounds.box_corners() - center
    from ..geometry import inner_product, Place
    d = max(inner_product(n,c) for c in bc for n in face_normals)
    d *= 1/max(0.01, 1-pad)
    view_direction = -position.z_axis()
    camera_center = center - d*view_direction
    va_position = Place(axes = position.axes(), origin = camera_center)
    return va_position

def perspective_view_width(point, origin, field_of_view):
    '''
    Return the visible width at the distance to the given point
    in scene coordinates.
    '''
    from ..geometry import vector
    d = vector.distance(origin, point)
    from math import radians
    # view width at center
    vw = d * radians(field_of_view)
    return vw

def perspective_projection_matrix(field_of_view, window_size, near_far_clip, pixel_shift):
    '''4x4 perspective projection matrix viewing along -z axis.'''
    from math import radians, tan
    fov = radians(field_of_view)
    near, far = near_far_clip
    w = 2 * near * tan(0.5 * fov)
    ww, wh = window_size
    aspect = wh / ww
    h = w * aspect
    left, right, bot, top = -0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h
    xps, yps = pixel_shift
    xshift, yshift = xps/ww, yps/wh
    pm = frustum(left, right, bot, top, near, far, xshift, yshift)
    return pm

def perspective_direction(window_x, window_y, window_size, field_of_view):
    '''
    Return points in camera coordinates at a given window pixel position
    at specified z depths.  Field of view is in degrees.
    '''
    from math import radians, tan
    fov = radians(field_of_view)
    t = tan(0.5 * fov)		# Field of view is in width
    wp, hp = window_size        # Screen size in pixels
    wx, wy = 2*(window_x - 0.5 * wp) / wp, 2*(0.5 * hp - window_y) / wp
    from ..geometry import normalize_vector
    d = normalize_vector((t*wx, t*wy, -1))
    return d

class OrthographicCamera(Camera):
    '''Orthographic projection camera.'''

    name = 'orthographic'

    def __init__(self, field_width = None):
        Camera.__init__(self)
        self.field_width = 1 if field_width is None or field_width <= 0 else field_width
        "Horizontal field width in scene coordinate units."

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that shifts the camera to show the bounding box.
        The camera view direction is not changed.
        '''
        p = self.position
        corners = p.inverse() * bounds.box_corners()	# In camera coords
        from ..geometry import point_bounds
        b = point_bounds(corners)
        xsize, ysize, zsize = b.xyz_max - b.xyz_min
        w = max(xsize, ysize * window_size[0] / window_size[1]) if window_size else xsize
        w *= 1/max(0.01, 1-pad)
        self.field_width = w if w > 0 else 1.0
        ca = bounds.center() - zsize*self.view_direction()
        shift = ca - self.position.origin()
        from ..geometry import translation
        self.position = translation(shift) * self.position

    def view_width(self, center):
        '''Return the width of the view at position center which is in
        scene coordinates.'''
        return self.field_width

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        ww, wh = window_size
        aspect = wh / ww
        w = self.field_width
        h = aspect * w
        left, right, bot, top = -0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h
        near, far = near_far_clip
        xps, yps = self._pixel_shift		# supersampling shift
        vxs, vys = self.view_pixel_shift(view_num)	# Per-view shift
        xshift, yshift = (xps+vxs)/ww, (yps+vys)/wh
        pm = ortho(left, right, bot, top, near, far, xshift, yshift)
        return pm

    def ray(self, window_x, window_y, window_size):
        '''
        Return origin and direction in scene coordinates of sight line
        for the specified window pixel position.
        '''
        wp, hp = window_size     # Screen size in pixels
        s = self.field_width
        x, y = s*(window_x - 0.5 * wp) / wp, s*(0.5 * hp - window_y) / wp
        p = self.position
        origin = p * (x,y,0)
        d = -p.z_axis()
        return (origin, d)

class StereoCamera(Camera):
    '''
    Sequential stereo camera mode.
    Uses two parameters, the eye spacing in scene units, and also
    the eye spacing in pixels in the window.  The two eyes are considered
    2 views that belong to one camera.
    '''
    name = 'stereo'

    def __init__(self, eye_separation_pixels=200):
        Camera.__init__(self)

        self.field_of_view = 30
        "Horizontal field of view in degrees."

        self.eye_separation_scene = 5.0
        "Stereo eye separation in scene units."

        self.eye_separation_pixels = eye_separation_pixels
        "Separation of the user's eyes in screen pixels used for stereo rendering."

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            s = -1 if view_num == 0 else 1
            es = self.eye_separation_scene
            from ..geometry import place
            t = place.translation((s * 0.5 * es, 0, 0))
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera mode.'''
        return 2

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  The camera view direction is not changed.
        '''
        self.position = perspective_view_all(bounds, self.position, self.field_of_view, window_size, pad)
        if window_size is not None:
            self._set_eye_separation(bounds.center(), window_size[0])

    def _set_eye_separation(self, point_on_screen, window_width):
        from ..geometry import inner_product
        z = inner_product(self.view_direction(), point_on_screen - self.position.origin())
        if z <= 0:
            return
        from math import tan, radians
        screen_width = 2*z*tan(0.5*radians(self.field_of_view))
        es = screen_width * self.eye_separation_pixels / window_width
        self.eye_separation_scene = es
        self.redraw_needed = True

    def view_width(self, center):
        '''Return the width of the view at position center which is in
        scene coordinates.'''
        return perspective_view_width(center, self.position.origin(), self.field_of_view)

    def view_pixel_shift(self, view_num):
        '''Shift of center away from center of render target.'''
        if view_num is None:
            return 0, 0
        s = -1 if view_num == 0 else 1
        return (s * 0.5 * self.eye_separation_pixels, 0)

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        render.set_stereo_buffer(view_num)

class SplitStereoCamera(Camera):
    '''Side-by-side and top-bottom stereo.'''

    name = 'split stereo'

    def __init__(self, layout = 'side-by-side'):

        Camera.__init__(self)
        self.field_of_view = 30				# Horizontal field, degrees
        self.eye_separation_scene = 5.0			# Angstroms
        self._framebuffer = {'left':None, 'right':None} # Framebuffer for rendering each eye
        self._drawing = {'left':None, 'right':None}	# Drawing of rectangle with cube map texture
        self.layout = layout			# Packing of left/right eye images: top-bottom or side-by-side

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame for a specific camera view number.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            # Stereo eyes view in same direction with position shifted along x.
            s = -1 if view_num == 0 else 1
            es = self.eye_separation_scene
            from ..geometry import place
            t = place.translation((s*0.5*es,0,0))
            v = camera_position * t
        return v

    def number_of_views(self):
        '''Number of views rendered by camera mode.'''
        return 2

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  The camera view direction is not changed.
        '''
        self.position = perspective_view_all(bounds, self.position, self.field_of_view, window_size, pad)

    def view_width(self, point):
        return perspective_view_width(point, self.position.origin(), self.field_of_view)

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        if view_num > 0:
            render.pop_framebuffer()	        # Pop left eye framebuffer
        eye = 'left' if view_num == 0 else 'right'
        fb = self._eye_framebuffer(eye, render.render_size())
        render.push_framebuffer(fb)		# Push eye framebuffer

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''
        render.pop_framebuffer()	        # Pop the right eye framebuffer.
        drawings = [self._eye_drawing(eye) for eye in ('left', 'right')]
        from .drawing import draw_overlays
        draw_overlays(drawings, render)

    def _eye_framebuffer(self, eye, window_size):
        fb = self._framebuffer[eye]
        w, h = window_size
        if self.layout == 'side-by-side':
            tw, th = w//2, h
        else:
            tw, th = w, h//2
        if fb is None or (tw, th) != (fb.width, fb.height):
            from .opengl import Texture, Framebuffer
            t = Texture()
            t.initialize_rgba((tw,th))
            fb = Framebuffer(color_texture = t)
            self._framebuffer[eye] = fb
            d = self._drawing[eye]
            if d:
                d.texture = fb.color_texture	# Update drawing texture
        return fb

    def _eye_drawing(self, eye):
        d = self._drawing[eye]
        if d is None:
            from .drawing import Drawing
            self._drawing[eye] = d = Drawing('%s eye' % eye)
            from numpy import array, float32, int32
            va = array(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)), float32)
            ta = array(((0,1,2),(0,2,3)), int32)
            tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
            if self.layout == 'top-bottom':
                # Shift left eye to top half of window, right eye to bottom half
                y = va[:,1]
                y[:] += (1 if eye == 'left' else -1)
                y[:] /= 2
            elif self.layout == 'side-by-side':
                # Shift left eye to left half of window, right eye to right half
                x = va[:,0]
                x[:] += (-1 if eye == 'left' else 1)
                x[:] /= 2
            d.geometry = va, ta
            d.color = (255,255,255,255)
            d.use_lighting = False
            d.texture_coordinates = tc
            d.texture = self._framebuffer[eye].color_texture
            d.opaque_texture = True
        return d

# glFrustum() matrix
def frustum(left, right, bottom, top, z_near, z_far, xshift=0, yshift=0):
    '''
    Return a 4 by 4 perspective projection matrix.  It includes a
    shift along x used to superpose offset left and right eye views in
    sequential stereo mode.
    '''
    a = (right + left) / (right - left) - 2 * xshift
    b = (top + bottom) / (top - bottom) - 2 * yshift
    c = - (z_far + z_near) / (z_far - z_near)
    d = - (2 * z_far * z_near) / (z_far - z_near)
    e = 2 * z_near / (right - left)
    f = 2 * z_near / (top - bottom)
    m = ((e, 0, 0, 0),
         (0, f, 0, 0),
         (a, b, c, -1),
         (0, 0, d, 0))
    return m


# glOrtho() matrix
def ortho(left, right, bottom, top, z_near, z_far, xshift=0, yshift=0):
    '''
    Return a 4 by 4 orthographic projection matrix.  It includes a
    shift along x used to superpose offset left and right eye views in
    sequential stereo mode.
    '''
    a = 1 / (right - left)
    b = left + right
    c = 1 / (top - bottom)
    d = bottom + top
    e = 1 / (z_far - z_near)
    f = z_near + z_far
    m = ((2 * a, 0, 0, 0),
         (0, 2 * c, 0, 0),
         (0, 0, -2 * e, 0),
         (- b * a + xshift, - d * c + yshift, - f * e, 1))
    return m


def camera_framing_drawings(drawings):
    '''
    Create a Camera object for viewing the specified models.
    This is used for capturing thumbnail images.
    '''
    c = MonoCamera()
    from ..geometry import bounds
    b = bounds.union_bounds(d.bounds() for d in drawings)
    if b is None:
        return None
    c.view_all(b)
    return c
