class Camera:
    '''
    A Camera has a position in the scene and viewing direction given by a Place object.
    The -z axis of the coordinate frame given by the Place object is the view direction.
    The x and y axes are the horizontal and vertical axes of the camera frame.
    A camera has an angular field of view measured in degrees.  It manages near and far
    clip planes.  In stereo modes it uses two additional parameters, the eye spacing
    in scene units, and also the eye spacing in pixels in the window.  The two eyes
    are considered 2 views that belong to one camera.
    '''
    def __init__(self, window_size, mode = 'mono', eye_separation_pixels = 200):

        self.window_size = window_size

        # Camera postion and direction, neg z-axis is camera view direction,
        # x and y axes are horizontal and vertical screen axes.
        # First 3 columns are x,y,z axes, 4th column is camara location.
        from ..geometry.place import Place
        self.place = self.place_inverse = Place()
        self.field_of_view = 45                   # degrees, width

        self.mode = mode                          # 'mono', 'stereo', 'oculus'
        self.eye_separation_scene = 1.0           # Scene distance units
        self.eye_separation_pixels = eye_separation_pixels        # Screen pixel units

        self.warp_window_size = window_size       # Used for scaling in oculus mode

        self.redraw_needed = False

    def initialize_view(self, center, size):
        '''
        Set the camera to completely show models having specified center and radius
        looking along the scene -z axis.
        '''
        cx,cy,cz = center
        from math import pi, tan
        fov = self.field_of_view*pi/180
        camdist = 0.5*size + 0.5*size/tan(0.5*fov)
        from ..geometry import place
        self.set_view(place.translation((cx,cy,cz+camdist)))

    def view_all(self, center, size):
        '''
        Return the shift that makes the camera completely show models having specified center and radius.
        The camera is not moved.
        '''
        from math import pi, tan
        fov = self.field_of_view*pi/180
        d = 0.5*size + 0.5*size/tan(0.5*fov)
        vd = self.view_direction()
        cp = self.position()
        from numpy import array, float32
        shift = array(tuple((center[a]-d*vd[a])-cp[a] for a in (0,1,2)), float32)
        return shift

    def view(self, view_num = None):
        '''
        Return the Place coordinate frame of the camera.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        m = self.mode
        if view_num is None or m == 'mono':
            v = self.place
        elif m == 'stereo' or m == 'oculus':
            # Stereo eyes view in same direction with position shifted along x.
            s = -1 if view_num == 0 else 1
            es = self.eye_separation_scene
            from ..geometry import place
            t = place.translation((s*0.5*es,0,0))
            v = self.place * t
        else:
            raise ValueError('Unknown camera mode %s' % m)
        return v

    def view_inverse(self, view_num = None):
        '''
        Return the inverse transform of the Camera view mapping scene coordinates to camera coordinates.
        '''
        if view_num is None or self.mode == 'mono':
            v = self.place_inverse
        else:
            v = self.view(view_num).inverse()
        return v
                
    def set_view(self, place):
        '''
        Reposition the camera using the specified Place coordinate frame.
        '''
        self.place = place
        self.place_inverse = place.inverse()
        self.redraw_needed = True

    def view_width(self, center):
        '''Return the width of the view at position center which is in scene coordinates.'''
        cp = self.position()
        vd = self.view_direction()
        d = sum((center-cp)*vd)         # camera to center of models
        from math import tan, pi
        vw = 2*d*tan(0.5*self.field_of_view*pi/180)     # view width at center
        return vw

    def pixel_size(self, center):
        '''
        Return the size of a pixel in scene units for a point at position center.
        Center is given in scene coordinates and perspective projection is accounted for.
        '''
        # Pixel size at center
        w,h = self.window_size
        from math import pi, tan
        fov = self.field_of_view * pi/180

        c = self.position()
        from ..geometry import vector
        ps = vector.distance(c,center) * 2*tan(0.5*fov) / w
        return ps

    def position(self, view_num = None):
        '''The position of the camera in scene coordinates.'''
        return self.view(view_num).translation()

    def view_direction(self, view_num = None):
        '''The view direction of the camera in scene coordinates.'''
        return -self.view(view_num).z_axis()

    def projection_matrix(self, near_far_clip, view_num = None, win_size = None):
        '''The 4 by 4 OpenGL perspective projection matrix for rendering the scene using this camera view.'''
        # Perspective projection to origin with center of view along z axis
        from math import pi, tan
        fov = self.field_of_view*pi/180
        near,far = near_far_clip
        near_min = 0.001*(far - near) if far > near else 1
        near = max(near, near_min)
        if far <= near:
            far = 2*near
        w = 2*near*tan(0.5*fov)
        ww,wh = w0,h0 = self.window_size if win_size is None else win_size
        m = self.mode
        if m == 'oculus':
            # Only half of window width used per eye in oculus mode.
            ww,wh = self.warp_window_size
        aspect = float(wh)/ww
        h = w*aspect
        left, right, bot, top = -0.5*w, 0.5*w, -0.5*h, 0.5*h
        if m in ('stereo','oculus') and not view_num is None:
            s = -1 if view_num == 0 else 1
            esp = self.eye_separation_pixels
            xwshift = s*float(esp)/(0.5*w0)
        else:
            xwshift = 0
        pm = frustum(left, right, bot, top, near, far, xwshift)
        return pm

    def clip_plane_points(self, window_x, window_y, z_distances):
        '''
        Two scene points at the near and far clip planes at the specified window pixel position.
        The points are in the camera coordinate frame.
        '''
        from math import pi, tan
        fov = self.field_of_view*pi/180
        t = tan(0.5*fov)
        wp,hp = self.window_size     # Screen size in pixels
        wx,wy = window_x - 0.5*wp, -(window_y - 0.5*hp)
        cpts = []
        for z in z_distances:
            w = 2*z*t   # Screen width in scene units
            r = w/wp if wp != 0 else 0
            c = (r*wx, r*wy, -z)
            cpts.append(c)
        return cpts

    def number_of_views(self):
        '''Number of view points for this camera.  Stereo modes have 2 views for left and right eyes.'''
        m = self.mode
        if m == 'mono':
            n = 1
        elif m == 'stereo' or m == 'oculus':
            n = 2
        else:
            raise ValueError('Unknown camera mode %s' % m)
        return n

    def set_framebuffer(self, view_num, render):
        '''Set the OpenGL drawing buffer and view port to render the scene.'''
        m = self.mode
        from .. import draw
        if m == 'mono':
            render.set_mono_buffer()
        elif m == 'stereo':
            render.set_stereo_buffer(view_num)
        elif m == 'oculus':
            render.push_framebuffer(self.warping_framebuffer())
        else:
            raise ValueError('Unknown camera mode %s' % m)

    def warp_image(self, view_num, render):
        m = self.mode
        if m == 'oculus':
            render.pop_framebuffer()
            w,h = self.window_size
            if view_num == 0:
                render.draw_background()
                render.set_drawing_region(0,0,w//2,h)
            elif view_num == 1:
                render.set_drawing_region(w//2,0,w//2,h)
            coffset = 0.5*self.eye_separation_pixels/(w//2)
            if view_num == 0:
                coffset = -coffset
            render.warp_center = (0.5 + coffset, 0.5)
            return self.warping_surface(render)
        return None

    def warping_framebuffer(self):

        tw,th = self.warp_window_size
        fb = getattr(self, 'warp_framebuffer', None)
        if fb is None or fb.width != tw or fb.height != th:
            from .. import draw
            t = draw.Texture()
            t.initialize_rgba(tw,th)
            self.warp_framebuffer = fb = draw.Framebuffer(texture = t)
        return fb

    def warping_surface(self, render):

        if not hasattr(self, 'warp_surface'):
            from ..surface import Surface
            self.warp_surface = s = Surface('warp plane')
            p = s.new_piece()
            from numpy import array, float32, int32
            va = array(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)), float32)
            ta = array(((0,1,2),(0,2,3)), int32)
            tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
            p.geometry = va, ta
            p.color = (1,1,1,1)
            p.use_lighting = False
            p.texture_coordinates = tc
            p.use_radial_warp = True

        s = self.warp_surface
        p = s.surface_pieces()[0]
        p.texture = self.warp_framebuffer.texture

        return s

# glFrustum() matrix
def frustum(left, right, bottom, top, zNear, zFar, xwshift = 0):
    '''
    Return a 4 by 4 perspective projection matrix.  It includes a shift along x used
    to superpose offset left and right eye views in sequential stereo mode.
    '''
    A = (right + left) / (right - left) - xwshift
    B = (top + bottom) / (top - bottom)
    C = - (zFar + zNear) / (zFar - zNear)
    D = - (2 * zFar * zNear) / (zFar - zNear)
    E = 2 * zNear / (right - left)
    F = 2 * zNear / (top - bottom)
    m = ((E, 0, 0, 0),
         (0, F, 0, 0),
         (A, B, C, -1),
         (0, 0, D, 0))
    return m

def camera_framing_models(w, h, models):

    c = Camera((w,h))
    from ..geometry import bounds
    b = bounds.union_bounds(m.placed_bounds() for m in models)
    center, size = bounds.bounds_center_and_radius(b)
    c.initialize_view(center, size)
    return c

def camera_command(cmdname, args, session):

    from .commands import float_arg, floats_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('mono', no_arg),
               ('stereo', no_arg),
               ('oculus', no_arg),
               ('fieldOfView', float_arg),      # degrees, width
               ('eyeSeparation', float_arg),    # physical units
               ('screenWidth', float_arg),      # physical units
               ('sEyeSeparation', float_arg),   # scene units
               ('middleDistance', no_arg),      # Adjust scene eye sep so models at screen depth.
               ('depthScale', float_arg),       # Scale scene and pixel eye separations
               ('nearFarClip', floats_arg, {'allowed_counts':(2,)}),     # scene units
               ('report', no_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    camera(session, **kw)

def camera(session, mono = None, stereo = None, oculus = None, fieldOfView = None, 
           eyeSeparation = None, screenWidth = None, sEyeSeparation = None,
           middleDistance = False, depthScale = None, report = False):

    v = session.view
    c = v.camera
    
    if mono or stereo or oculus:
        mode = 'mono' if mono else ('stereo' if stereo else 'oculus')
        v.set_camera_mode(mode)
    if not fieldOfView is None:
        c.field_of_view = fieldOfView
        c.redraw_needed = True
    if not eyeSeparation is None or not screenWidth is None:
        if eyeSeparation is None or screenWidth is None:
            from .commands import CommandError
            raise CommandError('Must specify eyeSeparation and screenWidth, only ratio is used')
        c.eye_separation_pixels = (eyeSeparation / screenWidth) * v.screen().size().width()
        c.redraw_needed = True
    if not sEyeSeparation is None:
        c.eye_separation_scene = sEyeSeparation
        c.redraw_needed = True
    if middleDistance:
        center, s = session.bounds_center_and_width()
        wscene = c.view_width(center)
        wpixels = v.window_size[0]
        c.eye_separation_scene = wscene * c.eye_separation_pixels / wpixels
        c.redraw_needed = True
    if not depthScale is None:
        # This scales the apparent depth while leaving apparent distance to models the same.
        c.eye_separation_pixels *= depthScale
        c.eye_separation_scene *= depthScale
        c.redraw_needed = True
    if report:
        msg = ('Camera\n' +
               'position %.5g %.5g %.5g\n' % tuple(c.position()) +
               'view direction %.6f %.6f %.6f\n' % tuple(c.view_direction()) +
               'field of view %.5g degrees\n' % c.field_of_view +
               'window size %d %d\n' % tuple(c.window_size) +
               'mode %s\n' % c.mode +
               'eye separation pixels %.5g, scene %.5g' % (c.eye_separation_pixels, c.eye_separation_scene))
        session.show_info(msg)
        smsg = 'Camera mode %s, field of view %.4g degrees' % (c.mode, c.field_of_view)
        session.show_status(smsg)
