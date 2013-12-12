class Camera:

    def __init__(self, window_size, mode = 'mono'):

        self.window_size = window_size

        # Camera postion and direction, neg z-axis is camera view direction,
        # x and y axes are horizontal and vertical screen axes.
        # First 3 columns are x,y,z axes, 4th column is camara location.
        from ..geometry.place import Place
        self.place = self.place_inverse = Place()
        self.field_of_view = 45                   # degrees, width
        self.near_far_clip = (1,100)              # along -z in camera coordinates

        self.mode = mode                          # 'mono', 'stereo', 'oculus'
        self.eye_separation_scene = 1.0           # Scene distance units
        self.eye_separation_pixels = 200.0        # Screen pixel units

        self.redraw_needed = False

    def initialize_view(self, center, size):

        cx,cy,cz = center
        from math import pi, tan
        fov = self.field_of_view*pi/180
        camdist = 0.5*size + 0.5*size/tan(0.5*fov)
        from ..geometry import place
        self.set_view(place.translation((cx,cy,cz+camdist)))
        self.near_far_clip = (camdist - size, camdist + size)

    def view_all(self, center, size):

        from math import pi, tan
        fov = self.field_of_view*pi/180
        d = 0.5*size + 0.5*size/tan(0.5*fov)
        vd = self.view_direction()
        cp = self.position()
        shift = tuple((center[a]-d*vd[a])-cp[a] for a in (0,1,2))
        self.near_far_clip = (d - size, d + size)
        return shift

    def view(self, view_num = None):
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
        if view_num is None or self.mode == 'mono':
            v = self.place_inverse
        else:
            v = self.view(view_num).inverse()
        return v
                
    def set_view(self, place):

        self.place = place
        self.place_inverse = place.inverse()
        self.redraw_needed = True

    def view_width(self, center):

        cp = self.position()
        vd = self.view_direction()
        d = sum((center-cp)*vd)         # camera to center of models
        from math import tan, pi
        vw = 2*d*tan(0.5*self.field_of_view*pi/180)     # view width at center of models
        return vw

    def pixel_size(self, center):

        # Pixel size at center
        w,h = self.window_size
        from math import pi, tan
        fov = self.field_of_view * pi/180

        c = self.position()
        from ..geometry import vector
        ps = vector.distance(c,center) * 2*tan(0.5*fov) / w
        return ps

    def set_near_far_clip(self, center, size):

        cp = self.position()
        vd = self.view_direction()
        d = sum((center-cp)*vd)         # camera to center of models
        self.near_far_clip = (d - size, d + size)

    def shift_near_far_clip(self, dz):
        n,f = self.near_far_clip
        self.near_far_clip = (n+dz,f+dz)

    def position(self, view_num = None):
        return self.view(view_num).translation()

    def view_direction(self, view_num = None):
        return -self.view(view_num).z_axis()

    def projection_matrix(self, view_num = None, win_size = None):

        # Perspective projection to origin with center of view along z axis
        from math import pi, tan
        fov = self.field_of_view*pi/180
        near,far = self.near_far_clip
        near_min = 0.001*(far - near) if far > near else 1
        near = max(near, near_min)
        if far <= near:
            far = 2*near
        w = 2*near*tan(0.5*fov)
        ww,wh = self.window_size if win_size is None else win_size
        aspect = float(wh)/ww
        m = self.mode
        if m == 'oculus':
            aspect *= 2
        h = w*aspect
        left, right, bot, top = -0.5*w, 0.5*w, -0.5*h, 0.5*h
        if m == 'stereo' and not view_num is None:
            s = -1 if view_num == 0 else 1
            esp = self.eye_separation_pixels
            xwshift = s*0.5*esp/ww
        else:
            xwshift = 0
        pm = frustum(left, right, bot, top, near, far, xwshift)
        return pm

    # Returns camera coordinates.
    def camera_clip_plane_points(self, window_x, window_y):
        znear, zfar = self.near_far_clip
        from math import pi, tan
        fov = self.field_of_view*pi/180
        wn = 2*znear*tan(0.5*fov)   # Screen width in model units, near clip
        wf = 2*zfar*tan(0.5*fov)    # Screen width in model units, far clip
        wp,hp = self.window_size     # Screen size in pixels
        rn, rf = (wn/wp, wf/wp) if wp != 0 else (1,1)
        wx,wy = window_x - 0.5*wp, -(window_y - 0.5*hp)
        cn = (rn*wx, rn*wy, -znear)
        cf = (rf*wx, rf*wy, -zfar)
        return cn, cf

    # Returns scene coordinates.
    def clip_plane_points(self, window_x, window_y):
        cn, cf = self.camera_clip_plane_points(window_x, window_y)
        mn, mf = self.place * (cn,cf)
        return mn, mf

    def number_of_views(self):
        m = self.mode
        if m == 'mono':
            n = 1
        elif m == 'stereo' or m == 'oculus':
            n = 2
        else:
            raise ValueError('Unknown camera mode %s' % m)
        return n

    def setup(self, view_num, render):
        m = self.mode
        from .. import draw
        if m == 'mono':
            render.set_mono_buffer()
        elif m == 'stereo':
            render.set_stereo_buffer(view_num)
        elif m == 'oculus':
            render.set_mono_buffer()
            w,h = self.window_size
            if view_num == 0:
                render.set_drawing_region(0,0,w//2,h)
            elif view_num == 1:
                render.set_drawing_region(w//2,0,w//2,h)
        else:
            raise ValueError('Unknown camera mode %s' % m)

# glFrustum() matrix
def frustum(left, right, bottom, top, zNear, zFar, xwshift = 0):
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
