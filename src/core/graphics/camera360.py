from .camera import CameraMode
class Mono360CameraMode(CameraMode):

    def __init__(self):

        self._framebuffer = None        	# Framebuffer for rendering each face
        self._cube_face_size = 1024		# Pixels
        self._projection_size = (360,180)	# Grid size for projecting cubemap.
        self._drawing = None			# Drawing of rectangle with cube map texture
        self._view_rotations = _cube_map_face_views()   # Camera views for cube faces

    def name(self):
        '''Name of camera mode.'''
        return 'mono 360'

    def set_camera_mode(self, camera):
        camera.field_of_view = 90
        camera.mode = self

    def view(self, camera_position, view_num):
        '''
        Return the Place coordinate frame for a specific camera view number.
        As a transform it maps camera coordinates to scene coordinates.
        '''
        v = camera_position
        if view_num is not None:
            v = v * self._view_rotations[view_num]
        return v

    def number_of_views(self):
        '''Number of views rendered by camera mode.'''
        return 6

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        fb = self._cube_face_framebuffer()
        if view_num == 0:
            render.push_framebuffer(fb)
        fb.set_cubemap_face(view_num)
        _adjust_light_directions(render, self._view_rotations[view_num])

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''
        _adjust_light_directions(render)	# Restore light directions
        render.pop_framebuffer()	        # Pop the cube face framebuffer.
        cubemap = self._cube_face_framebuffer().color_texture
        proj = self._projection_drawing()
        _project_cubemap(cubemap, proj, render) # Project cubemap to longitude/lattitude rectangle

    def _cube_face_framebuffer(self):
        fb = self._framebuffer
        if fb is None:
            self._framebuffer = fb = _cube_map_framebuffer(self._cube_face_size)
        return fb

    def _projection_drawing(self):
        d = self._drawing
        if d is None:
            self._drawing = d = _equirectangular_projection_drawing(self._projection_size)
        return d

# Camera rotations for 6 cube faces. Face order +x,-x,+y,-y,+z,-z.
def _cube_map_face_views():
    from ..geometry import Place
    views = [Place(matrix=m) for m in
             (((0,0,-1,0),(0,-1,0,0),(-1,0,0,0)),
              ((0,0,1,0),(0,-1,0,0),(1,0,0,0)),
              ((1,0,0,0),(0,0,-1,0),(0,1,0,0)),
              ((1,0,0,0),(0,0,1,0),(0,-1,0,0)),
              ((1,0,0,0),(0,-1,0,0),(0,0,-1,0)),
              ((-1,0,0,0),(0,-1,0,0),(0,0,1,0)))]
    return views

def _cube_map_framebuffer(size):
    from . import Texture, opengl
    t = Texture(cube_map = True)
    t.initialize_rgba((size,size))
    fb = opengl.Framebuffer(color_texture = t)
    return fb

# Project cubemap to longitude/lattitude rectangle
def _project_cubemap(cubemap_texture, projection_drawing, render):
    projection_drawing.texture = cubemap_texture
    from .drawing import draw_overlays
    draw_overlays([projection_drawing], render)

def _adjust_light_directions(render, rotation = None):
    l = render.lighting
    if not hasattr(l, '_original_key_light_direction'):
        l._original_key_light_direction = l.key_light_direction
        l._original_fill_light_direction = l.fill_light_direction
    if rotation is None:
        l.key_light_direction = l._original_key_light_direction
        l.fill_light_direction = l._original_fill_light_direction
        delattr(l, '_original_key_light_direction')
        delattr(l, '_original_fill_light_direction')
    else:
        rinv = rotation.inverse()
        l.key_light_direction = rinv * l._original_key_light_direction
        l.fill_light_direction = rinv * l._original_fill_light_direction
    render.set_shader_lighting_parameters()

def _equirectangular_projection_drawing(size):
    w,h = size

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
            tc[j,i,:] = (-sa[i]*sb[j],-cb[j],ca[i]*sb[j])     # z-axis in middle of rectangle
    tc = tc.reshape((h*w,3))

    # Create rectangle drawing with sphere point texture coordinates.
    from . import Drawing
    d = Drawing('equirectangular projection')
    d.geometry = va, ta
    d.color = (255,255,255,255)
    d.use_lighting = False
    d.texture_coordinates = tc

    return d

mono_360_camera_mode = Mono360CameraMode()
