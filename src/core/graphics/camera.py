# vi: set expandtab shiftwidth=4 softtabstop=4:
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

    def name(self):
        '''Name indicating the type of camera, for example, "mono", "stereo", "orthographic".'''
        return 'mono'

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

    def pixel_shift(self, view_num):
        '''
        Per view pixel shift of center away from center of render target.
        This is used for example to shift stereoscopic left/right eye images.
        '''
        return (0,0)

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
        return

    def do_swap_buffers(self):
        return True

    def get_position(self, view_num=None):
        p = self._position if view_num is None else self.view(self._position, view_num)
        return p

    def set_position(self, p):
        self._position = p
        self.redraw_needed = True
    position = property(get_position, set_position)
    '''Location and orientation of the camera in scene coordinates. Camera
    points along -z axis.'''

    def initialize_view(self, center, size):
        '''
        Set the camera position to completely show models having specified center
        and radius looking along the scene -z axis.
        '''
        pass

    def view_all(self, center, size):
        '''
        Return the shift that makes the camera completely show models
        having specified center and radius.  The camera view direction
        is not changed.
        '''
        raise NotImplementedError('Camera has no view_all() method')

    def view_width(self, center):
        '''Return the width of the view at position center which is in
        scene coordinates.'''
        return None

    def pixel_size(self, center, window_size):
        '''
        Return the size of a pixel in scene units for a point at position
        center.  Center is given in scene coordinates and perspective
        projection is accounted for.
        '''
        raise NotImplementedError('Camera has no pixel_size() method')

    def view_direction(self, view_num=None):
        '''The view direction of the camera in scene coordinates.'''
        return -self.get_position(view_num).z_axis()

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        xps, yps = self._pixel_shift		# supersampling shift
        vxs, vys = self.pixel_shift(view_num)	# Per-view shift
        pixel_shift = (xps + vxs, yps + vys)
        return perspective_projection_matrix(self.field_of_view, window_size,
                                             near_far_clip, pixel_shift)

    def clip_plane_points(self, window_x, window_y, window_size, z_distances):
        '''
        Return two scene points at the near and far clip planes at the
        specified window pixel position.
        '''
        return [None]*len(z_distances)

class MonoCamera(Camera):
    '''Perspective projection camera has an angular field of view measured in degrees.'''
    def __init__(self):
        Camera.__init__(self)
        self.field_of_view = 45
        "Horizontal field of view in degrees."

    def initialize_view(self, center, size):
        '''
        Set the camera position to completely show models having specified center
        and radius looking along the scene -z axis.
        '''
        view_direction = (0,0,-1)
        ca = perspective_view_all_center(center, size, view_direction, self.field_of_view)
        from ..geometry import translation
        self.position = translation(ca)

    def view_all(self, center, size):
        '''
        Return the shift that makes the camera completely show models
        having specified center and radius.  The camera view direction
        is not changed.
        '''
        ca = perspective_view_all_center(center, size, self.view_direction(), self.field_of_view)
        return ca - self.position.origin()

    def view_width(self, center):
        '''Return the width of the view at position center which is in
        scene coordinates.'''
        return perspective_view_width(center, self.position.origin(),
                                      self.view_direction(), self.field_of_view)

    def pixel_size(self, center, window_size):
        '''
        Return the size of a pixel in scene units for a point at position
        center.  Center is given in scene coordinates and perspective
        projection is accounted for.
        '''
        return perspective_pixel_size(center, self.position.origin(),
                                      self.field_of_view, window_size[0])

    def clip_plane_points(self, window_x, window_y, window_size, z_distances):
        '''
        Return scene points at the near and far clip planes at the
        specified window pixel position.
        '''
        cpts = perspective_clip_plane_points(window_x, window_y, window_size,
                                             self.field_of_view, z_distances)
        spts = self.position * cpts  # Convert camera to scene coordinates
        return spts

def perspective_view_all_center(center, size, direction, field_of_view):
    '''
    Return the camera position that shows models with bounds specified by
    a center and radius. Camera has perspective projection.
    '''
    from math import radians, tan
    fov = radians(field_of_view)
    d = 0.5 * size + 0.5 * size / tan(0.5 * fov)
    from numpy import array, float32
    center = array(tuple((center[a] - d * direction[a]) for a in (0, 1, 2)), float32)
    return center

def perspective_view_width(point, origin, view_direction, field_of_view):
    '''
    Return the visible width based on the distance to the given point
    in scene coordinates.
    '''
    d = sum((point - origin) * view_direction)   # camera distance to point
    from math import radians, tan
    # view width at center
    vw = 2 * d * tan(0.5 * radians(field_of_view))
    return vw

def perspective_pixel_size(point, origin, field_of_view, window_width):
    '''Pixel width in scene units and point.'''
    from math import radians, tan
    fov = radians(field_of_view)
    from ..geometry import vector
    ps = vector.distance(origin, point) * 2 * tan(0.5 * fov) / window_width
    return ps

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

def perspective_clip_plane_points(window_x, window_y, window_size,
                                  field_of_view, z_distances):
    '''
    Return points in camera coordinates at a given window pixel position
    at specified z depths.
    '''
    from math import radians, tan
    fov = radians(field_of_view)
    t = tan(0.5 * fov)
    wp, hp = window_size     # Screen size in pixels
    wx, wy = (window_x - 0.5 * wp) / wp, (0.5 * hp - window_y) / wp
    cpts = []
    for z in z_distances:
        w = 2 * z * t   # Render width in scene units
        c = (w * wx, w * wy, -z)
        cpts.append(c)
    return cpts

class StereoCamera(Camera):
    '''
    Sequential stereo camera mode.
    Uses two parameters, the eye spacing in scene units, and also
    the eye spacing in pixels in the window.  The two eyes are considered
    2 views that belong to one camera.
    '''

    def __init__(self, eye_separation_pixels=200):
        self.eye_separation_scene = 1.0
        "Stereo eye separation in scene units."

        self.eye_separation_pixels = eye_separation_pixels
        """Separation of the user's eyes in screen pixels used for stereo
        rendering."""

    def name(self):
        return 'stereo'

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

    def pixel_shift(self, view_num):
        '''Shift of center away from center of render target.'''
        if view_num is None:
            return 0, 0
        s = -1 if view_num == 0 else 1
        return (s * 0.5 * self.eye_separation_pixels, 0)

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        render.set_stereo_buffer(view_num)

stereo_camera = StereoCamera()


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
    c = Camera()
    from ..geometry import bounds
    b = bounds.union_bounds(d.bounds() for d in drawings)
    if b is None:
        return None
    c.initialize_view(b.center(), b.width())
    return c
