from .camera import CameraMode
class CubeMapCameraMode(CameraMode):

    def __init__(self):

        self._framebuffer = None        	# Framebuffer for rendering each face
        self._cube_face_size = 1024		# Pixels
        self._projection_subdivision = 1	# Degrees, separation of interpolation points
        self._drawing = None			# Drawing of rectangle with cube map texture

        # Camera rotations for 6 cube faces. Face order +x,-x,+y,-y,+z,-z.
        from ..geometry import Place
        self._view_rotations = [Place(matrix=m) for m in
                                (((0,0,-1,0),(0,-1,0,0),(-1,0,0,0)),
                                 ((0,0,1,0),(0,-1,0,0),(1,0,0,0)),
                                 ((1,0,0,0),(0,0,-1,0),(0,1,0,0)),
                                 ((1,0,0,0),(0,0,1,0),(0,-1,0,0)),
                                 ((1,0,0,0),(0,-1,0,0),(0,0,-1,0)),
                                 ((-1,0,0,0),(0,-1,0,0),(0,0,1,0)))]

    def name(self):
        '''Name of camera mode.'''
        return 'equirectangular'

    def set_camera_mode(self, camera):
        camera.field_of_view = 90
        camera.mode = self

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame for a specific camera view number.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        if view_num is None:
            v = camera_position
        else:
            v = camera_position * self._view_rotations[view_num]
        return v

    def number_of_views(self):
        '''Number of views rendered by camera mode.'''
        return 6

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        if view_num == 0:
            render.push_framebuffer(self._cube_face_framebuffer())
        self._set_texture_face(view_num)
        self._adjust_light_directions(view_num, render)

    def _adjust_light_directions(self, view_num, render):
        l = render.lighting
        if not hasattr(l, '_original_key_light_direction'):
            l._original_key_light_direction = l.key_light_direction
            l._original_fill_light_direction = l.fill_light_direction
        if view_num is None:
            l.key_light_direction = l._original_key_light_direction
            l.fill_light_direction = l._original_fill_light_direction
            delattr(l, '_original_key_light_direction')
            delattr(l, '_original_fill_light_direction')
        else:
            rinv = self._view_rotations[view_num].inverse()
            l.key_light_direction = rinv * l._original_key_light_direction
            l.fill_light_direction = rinv * l._original_fill_light_direction
        render.set_shader_lighting_parameters()

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''

        self._adjust_light_directions(None, render)	# Restore light directions

        render.pop_framebuffer()        # Pop the cube face framebuffer.

        d = self._projection_drawing()
        d.texture = self._cube_face_framebuffer().color_texture
        from .drawing import draw_overlays
        draw_overlays([d], render)

    def _set_texture_face(self, view_num):
        fb = self._cube_face_framebuffer()
        fb.set_cubemap_face(view_num)
        return fb

    def _cube_face_framebuffer(self):
        fb = self._framebuffer
        if fb is None:
            s = self._cube_face_size
            from . import Texture, opengl
            t = Texture(cube_map = True)
            t.initialize_rgba((s,s))
            self._framebuffer = fb = opengl.Framebuffer(color_texture = t)
        return fb

    def _projection_drawing(self):

        d = self._drawing
        if d:
            return d

        psd = self._projection_subdivision
        w = max(4,int(360.0/psd))
        h = w//2

        # Compute vertices (-1 to 1 range) for rectangular grid.
        from numpy import arange, empty, float32, int32, cos, sin
        x = arange(w)*(2/(w-1)) - 1
        y = arange(h)*(2/(h-1)) - 1
        va = empty((h,w,3), float32)
        for i in range(w):
            va[:,i,1] = y
        for j in range(h):
            va[j,:,0] = x
        va[:,:,2] = 0
        va = va.reshape((h*w,3))

        # Compute triangles for rectangular grid
        ta = empty((h-1,w-1,2,3), int32)
        for j in range(h-1):
            for i in range(w-1):
                ta[j,i,0,:] = (j*w+i, j*w+(i+1), (j+1)*w+(i+1))
                ta[j,i,1,:] = (j*w+i, (j+1)*w+(i+1), (j+1)*w+i)
        ta = ta.reshape(((h-1)*(w-1)*2, 3))

        # Compute direction vectors as texture coordinates
        from math import pi
        a = arange(w)*(2*pi/w)
        ca, sa = cos(a), sin(a)
        b = arange(h)*(pi/h)
        cb, sb = cos(b), sin(b)
        tc = empty((h,w,3), float32)
        for j in range(h):
            for i in range(w):
#                tc[j,i,:] = (ca[i]*sb[j],sa[i]*sb[j],cb[j])
                tc[j,i,:] = (-sa[i]*sb[j],-cb[j],ca[i]*sb[j])     # z-axis in middle of rectangle
        tc = tc.reshape((h*w,3))

        # Create rectangle drawing with sphere point texture coordinates.
        from . import Drawing
        self._drawing = d = Drawing('equirectangular projection')
        d.geometry = va, ta
        d.color = (255,255,255,255)
        d.use_lighting = False
        d.texture_coordinates = tc

        return d

equirect_camera_mode = CubeMapCameraMode()
