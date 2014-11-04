'''
Camera
======
'''

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
    def __init__(self, mode = 'mono', eye_separation_pixels = 200):

        # Camera postion and direction, neg z-axis is camera view direction,
        # x and y axes are horizontal and vertical screen axes.
        # First 3 columns are x,y,z axes, 4th column is camara location.
        from ..geometry.place import Place
        self.place = self.place_inverse = Place()
        self.field_of_view = 45                   # degrees, width

        self.mode = mode                          # 'mono', 'stereo', 'oculus'
        self.eye_separation_scene = 1.0           # Scene distance units
        self.eye_separation_pixels = eye_separation_pixels        # Screen pixel units
        self.oculus_centering_shift = 0,0               # pixels, left eye

        self.pixel_shift = 0,0                   # x and y shift for supersampling

        self.warp_framebuffers = [None, None]   # Off-screen rendering each eye for Oculus Rift
        self.warp_window_size = None

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

    def pixel_size(self, center, window_size):
        '''
        Return the size of a pixel in scene units for a point at position center.
        Center is given in scene coordinates and perspective projection is accounted for.
        '''
        # Pixel size at center
        w,h = window_size
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

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL perspective projection matrix for rendering the scene using this camera view.'''
        # Perspective projection to origin with center of view along z axis
        from math import pi, tan
        fov = self.field_of_view*pi/180
        near,far = near_far_clip
        w = 2*near*tan(0.5*fov)
        ww,wh = window_size
        aspect = wh/ww
        h = w*aspect
        left, right, bot, top = -0.5*w, 0.5*w, -0.5*h, 0.5*h
        xps,yps = self.pixel_shift
        xshift,yshift = xps/ww, yps/wh
        if self.mode == 'stereo' and not view_num is None:
            s = -1 if view_num == 0 else 1
            xshift += s*0.5*self.eye_separation_pixels/ww
        if self.mode == 'oculus' and not view_num is None:
            s = 1 if view_num == 0 else -1
            sx,sy = self.oculus_centering_shift # For left eye
            xshift += s*sx/ww
            yshift += s*sy/wh

        pm = frustum(left, right, bot, top, near, far, xshift, yshift)
        return pm

    def clip_plane_points(self, window_x, window_y, window_size, z_distances, render):
        '''
        Return two scene points at the near and far clip planes at the specified window pixel position.
        '''
        from math import pi, tan
        fov = self.field_of_view*pi/180
        t = tan(0.5*fov)
        if self.mode == 'oculus':
            wx,wy,view_num = self.unwarped_render_position(window_x, window_y, window_size, render)
        else:
            wp,hp = window_size     # Screen size in pixels
            wx,wy,view_num = (window_x - 0.5*wp)/wp, (0.5*hp - window_y)/wp, 0
        cpts = []
        for z in z_distances:
            w = 2*z*t   # Render width in scene units
            c = (w*wx, w*wy, -z)
            cpts.append(c)
        spts = self.view(view_num) * cpts       # Convert camera to scene coordinates
        return spts

    def unwarped_render_position(self, window_x, window_y, window_size, render):
        # TODO: This would be very hard using the Oculus SDK distortion rendering.
        #       Used to do this when my code applied distortion.  This is needed for mouse selection.
        wx,wy,view_num = 0,0,0
        return wx,wy,view_num

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
        if m == 'mono':
            render.set_mono_buffer()
        elif m == 'stereo':
            render.set_stereo_buffer(view_num)
        elif m == 'oculus':
            if view_num > 0:
                render.pop_framebuffer()
            render.push_framebuffer(self.warping_framebuffer(view_num))
        else:
            raise ValueError('Unknown camera mode %s' % m)

    def warping_framebuffer(self, view_num):

        tw,th = self.warp_window_size
        
        fb = self.warp_framebuffers[view_num]
        if fb is None or fb.width != tw or fb.height != th:
            from . import opengl
            t = opengl.Texture()
            t.initialize_rgba((tw,th))
            self.warp_framebuffers[view_num] = fb = opengl.Framebuffer(color_texture = t)
            print ('created warp texture', t.size[0], t.size[1])
        return fb

    def draw_warped(self, render, session):
        '''Combine left and right eye images from separate textures with warping.'''

        m = self.mode
        if m != 'oculus':
            return

        render.pop_framebuffer()
        fb = render.current_framebuffer()
        render.draw_background()

        from . import opengl
        b = opengl.deactivate_bindings()  # Prevent oculus_render() from screwing up my last vertex array object.
        t0,t1 = [rb.color_texture for rb in self.warp_framebuffers]
        from ..devices import oculus
        oculus.oculus_render(t0.size[0], t0.size[1], t0.id, t1.id, session)

#        self.draw_unwarped(render)

    def draw_unwarped(self, render):

        # Unwarped rendering, for testing purposes.
        render.draw_background()

        # Draw left eye
        fb = render.current_framebuffer()
        w,h = fb.width//2, fb.height
        render.set_viewport(0,0,w,h)

        s = self.warping_surface(render)
        s.texture = self.warp_framebuffers[0].color_texture
        from . import drawing
        drawing.draw_overlays([s], render)

        # Draw right eye
        render.set_viewport(w,0,w,h)
        s.texture = self.warp_framebuffers[1].color_texture
        drawing.draw_overlays([s], render)

        session.view.swap_opengl_buffers()

    def warping_surface(self, render):

        if hasattr(self, 'warp_surface'):
            return self.warp_surface

        from . import Drawing
        self.warp_surface = s = Drawing('warp plane')
        # TODO: Use a childless drawing.
        from numpy import array, float32, int32
        va = array(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)), float32)
        ta = array(((0,1,2),(0,2,3)), int32)
        tc = array(((0,0),(1,0),(1,1),(0,1)), float32)
        s.geometry = va, ta
        s.color = (255,255,255,255)
        s.use_lighting = False
        s.texture_coordinates = tc
        return s

# glFrustum() matrix
def frustum(left, right, bottom, top, zNear, zFar, xshift = 0, yshift = 0):
    '''
    Return a 4 by 4 perspective projection matrix.  It includes a shift along x used
    to superpose offset left and right eye views in sequential stereo mode.
    '''
    A = (right + left) / (right - left) - 2*xshift
    B = (top + bottom) / (top - bottom) - 2*yshift
    C = - (zFar + zNear) / (zFar - zNear)
    D = - (2 * zFar * zNear) / (zFar - zNear)
    E = 2 * zNear / (right - left)
    F = 2 * zNear / (top - bottom)
    m = ((E, 0, 0, 0),
         (0, F, 0, 0),
         (A, B, C, -1),
         (0, 0, D, 0))
    return m

def camera_framing_models(models):

    c = Camera()
    from ..geometry import bounds
    b = bounds.union_bounds(m.bounds() for m in models)
    center, size = bounds.bounds_center_and_radius(b)
    if center is None:
        return None
    c.initialize_view(center, size)
    return c
