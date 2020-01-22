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

from .camera import Camera
class CubeMapCamera(Camera):

    name = 'cube map'

    def __init__(self, projection_size = (360, 180), cube_face_size = 1024, aspect = None):

        Camera.__init__(self)
        self._framebuffer = None        	# Framebuffer for rendering each face
        self._cube_face_size = cube_face_size	# Pixels
        self._projection_size = projection_size	# Grid size for projecting cubemap.
        self._aspect = aspect
        self._drawing = None			# Drawing of rectangle with cube map texture
        self._view_rotations = _cube_map_face_views()   # Camera views for cube faces

    def delete(self):
        fb = self._framebuffer
        if fb:
            fb.delete(make_current = True)
            self._framebuffer = None
            
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

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  The camera view direction is not changed.
        '''
        self.position = view_all_360(bounds, self.position)

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        from .camera import perspective_projection_matrix
        return perspective_projection_matrix(90, window_size, near_far_clip, self._pixel_shift)

    def set_special_render_modes(self, render):
        # Turn off depth cue since we don't support radial depth cueing.
        # Also don't have APIs to determine a near bound for radial depth cue
        # if camera is in a pocket surrounded by atoms, a typical 360 camera scenario.
        render.enable_capabilities &= ~render.SHADER_DEPTH_CUE

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        fb = self._cube_face_framebuffer(render.opengl_context)
        if view_num == 0:
            render.push_framebuffer(fb)
        fb.set_cubemap_face(view_num)
        _adjust_light_directions(render, self._view_rotations[view_num])

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''
        _adjust_light_directions(render)	# Restore light directions
        render.pop_framebuffer()	        # Pop the cube face framebuffer.
        cubemap = self._cube_face_framebuffer(render.opengl_context).color_texture
        proj = self._projection_drawing(render.render_size())
        _project_cubemap(cubemap, proj, render) # Project cubemap to longitude/lattitude rectangle

    def _cube_face_framebuffer(self, opengl_context):
        fb = self._framebuffer
        if fb is None:
            self._framebuffer = fb = _cube_map_framebuffer(opengl_context, self._cube_face_size)
        return fb

    def _projection_drawing(self, window_size):
        d = self._drawing
        if d is None:
            d = CubeMapProjectionDrawing(self._projection_size, self._direction_map)
            self._drawing = d
        if self._aspect is not None:
            d.set_aspect(self._aspect, window_size)
        return d

    def view_width(self, point):
        return view_width_360(point, self.position.origin())

    def _direction_map(self, x, y):
        '''
        Maps image x,y position (0-1 range) to a 3d direction vector
        for projecting the cube map onto a rectangle.
        Derived class must define this function.
        '''
        pass

class Mono360Camera(CubeMapCamera):
    name = 'mono 360'

    def __init__(self, projection_size = (360, 180), cube_face_size = 1024):
        CubeMapCamera.__init__(self, projection_size = projection_size,
                               cube_face_size = cube_face_size)
    
    def _direction_map(self, x, y):
        return _equirectangular_direction(x,y)

class DomeCamera(CubeMapCamera):
    name = 'dome'

    def __init__(self, projection_size = (180, 180), cube_face_size = 1024, aspect = 1):
        CubeMapCamera.__init__(self, projection_size = projection_size,
                               cube_face_size = cube_face_size, aspect = aspect)
        
    def _direction_map(self, x, y):
        return _fisheye_direction(x,y)
    
class Stereo360Camera(Camera):

    name = 'stereo 360'

    def __init__(self, layout = 'top-bottom', cube_face_size = 1024):

        Camera.__init__(self)
        self.eye_separation_scene = 0.2			# Angstroms
        self._framebuffer = {'left':None, 'right':None} # Framebuffer for rendering each face
        self._cube_face_size = cube_face_size		# Pixels
        self._projection_size = (360,180)		# Grid size for projecting cubemap.
        self._drawing = {'left':None, 'right':None}	# Drawing of rectangle with cube map texture
        v = _cube_map_face_views()
        self._view_rotations = v + v		# Camera views for cube faces
        self.layout = layout			# Packing of left/right eye images: top-bottom or side-by-side

    def delete(self):
        for fb in self._framebuffer.values():
            if fb:
                fb.delete(make_current = True)
        self._framebuffer = {}

        for d in self._drawing.values():
            d.delete()
        self._drawing = {}
        
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
        return 12

    def view_all(self, bounds, window_size = None, pad = 0):
        '''
        Return the shift that makes the camera completely show models
        having specified bounds.  The camera view direction is not changed.
        '''
        self.position = view_all_360(bounds, self.position)

    def view_width(self, point):
        return view_width_360(point, self.position.origin())

    def projection_matrix(self, near_far_clip, view_num, window_size):
        '''The 4 by 4 OpenGL projection matrix for rendering the scene.'''
        from .camera import perspective_projection_matrix
        return perspective_projection_matrix(90, window_size, near_far_clip, self._pixel_shift)

    def set_special_render_modes(self, render):
        render.enable_capabilities |= render.SHADER_STEREO_360
        # Turn off depth cue since we don't support radial depth cueing.
        render.enable_capabilities &= ~render.SHADER_DEPTH_CUE

    def clear_special_render_modes(self, render):
        render.enable_capabilities &= ~render.SHADER_STEREO_360

    def set_render_target(self, view_num, render):
        '''Set the OpenGL drawing buffer and viewport to render the scene.'''
        eye = 'left' if view_num < 6 else 'right'
        fb = self._cube_face_framebuffer(eye, render.opengl_context)
        if view_num == 0:
            render.push_framebuffer(fb)		# Push left eye framebuffer
            self._set_stereo_360_shader_parameters(render, eye)
        elif view_num == 6:
            render.pop_framebuffer()	        # Pop left eye framebuffer
            render.push_framebuffer(fb)		# Push right eye framebuffer
            self._set_stereo_360_shader_parameters(render, eye)
        fb.set_cubemap_face(view_num % 6)
        _adjust_light_directions(render, self._view_rotations[view_num])

    def _set_stereo_360_shader_parameters(self, render, eye):
        p = self.position
        es = self.eye_separation_scene
        xshift = -0.5*es if eye == 'left' else 0.5*es
        render.set_stereo_360_params(p.origin(), p.axes()[1], xshift)

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''
        _adjust_light_directions(render)	# Restore light directions
        render.pop_framebuffer()	        # Pop the cube face framebuffer.
        for eye in ('left', 'right'):
            cubemap = self._cube_face_framebuffer(eye, render.opengl_context).color_texture
            proj = self._projection_drawing(eye)
            _project_cubemap(cubemap, proj, render) # Project cubemap to longitude/lattitude rectangle

    def _cube_face_framebuffer(self, eye, opengl_context):
        fb = self._framebuffer[eye]
        if fb is None:
            self._framebuffer[eye] = fb = _cube_map_framebuffer(opengl_context, self._cube_face_size)
        return fb

    def _projection_drawing(self, eye):
        d = self._drawing[eye]
        if d is None:
            self._drawing[eye] = d = _equirectangular_projection_drawing(self._projection_size)
            if self.layout == 'top-bottom':
                # Shift left eye to top half of window, right eye to bottom half
                y = d.vertices[:,1]
                y[:] += (1 if eye == 'left' else -1)
                y[:] /= 2
            elif self.layout == 'side-by-side':
                # Shift left eye to left half of window, right eye to right half
                x = d.vertices[:,0]
                x[:] += (-1 if eye == 'left' else 1)
                x[:] /= 2

        return d

def view_width_360(point, origin):
    from math import pi
    from ..geometry import vector
    return 2 * pi * vector.distance(origin, point)

def view_all_360(bounds, cam_position):
    center, size = bounds.center(), bounds.width()
    shift = center - cam_position.origin() + 2*size*cam_position.z_axis()
    from ..geometry import translation
    return translation(shift) * cam_position

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

def _cube_map_framebuffer(opengl_context, size):
    from . import Texture, opengl
    t = Texture(cube_map = True)
    t.initialize_rgba((size,size))
    fb = opengl.Framebuffer('cubemap', opengl_context, color_texture = t)
    return fb

# Project cubemap to longitude/lattitude rectangle
def _project_cubemap(cubemap_texture, projection_drawing, render):
    if projection_drawing.pad():
        render.draw_background()
    dc = render.disable_capabilities
    render.disable_capabilities |= render.SHADER_STEREO_360
    d = projection_drawing
    d.texture = cubemap_texture
    d.opaque_texture = True
    from .drawing import draw_overlays
    draw_overlays([d], render)
    render.disable_capabilities = dc

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
    render.update_lighting_parameters()

def _equirectangular_direction(x, y):
    '''-z axis in middle of rectangle'''
    from math import pi, cos, sin
    theta, phi = x * 2*pi, y * pi
    ct, st = cos(theta), sin(theta)
    cp, sp = cos(phi), sin(phi)
    return (-st*sp,-cp,ct*sp)

def _fisheye_direction(x, y):
    '''-z axis in middle of rectangle'''
    xs, ys = 2*x - 1, 2*y - 1
    from math import atan2, pi, sqrt, cos, sin
    theta = atan2(ys, xs)
    phi = 0.5*pi*sqrt(xs*xs + ys*ys)
    ct, st = cos(theta), sin(theta)
    cp, sp = cos(phi), sin(phi)
    return (ct*sp, st*sp, -cp)

from . import Drawing
class CubeMapProjectionDrawing(Drawing):
    def __init__(self, size, direction_function):
        Drawing.__init__(self, 'cube map projection')

        w,h = size
        va = self._vertex_array(size)
        ta = self._triangle_array(size)
        tc = self._texture_coordinate_array(size, direction_function)
        
        # Create rectangle drawing with sphere point texture coordinates.
        self.set_geometry(va, None, ta)
        self.color = (255,255,255,255)
        self.use_lighting = False
        self.texture_coordinates = tc

        self._aspect = None
        self._window_size = None
        self._unit_vertices = va

    def _vertex_array(self, size):
        # Compute vertices (-1 to 1 range) for rectangular grid.
        from numpy import arange, empty, float32
        w,h = size
        x = arange(w)*(2/(w-1)) - 1
        y = arange(h)*(2/(h-1)) - 1
        va = empty((h,w,3), float32)
        for i in range(w):
            va[:,i,1] = y
        for j in range(h):
            va[j,:,0] = x
        va[:,:,2] = 0
        va = va.reshape((h*w,3))
        return va

    def _triangle_array(self, size):
        # Compute triangles for rectangular grid
        w,h = size
        from numpy import empty, int32
        ta = empty((h-1,w-1,2,3), int32)
        for j in range(h-1):
            for i in range(w-1):
                ta[j,i,0,:] = (j*w+i, j*w+(i+1), (j+1)*w+(i+1))
                ta[j,i,1,:] = (j*w+i, (j+1)*w+(i+1), (j+1)*w+i)
        ta = ta.reshape(((h-1)*(w-1)*2, 3))
        return ta

    def _texture_coordinate_array(self, size, direction_function):
        # Compute direction vectors as texture coordinates
        from numpy import empty, float32
        w,h = size
        tc = empty((h,w,3), float32)
        for j in range(h):
            for i in range(w):
                tc[j,i,:] = direction_function((i+.5)/w, (j+.5)/h)
        tc = tc.reshape((h*w,3))
        return tc

    def set_aspect(self, aspect, window_size):
        if  self._aspect == aspect and self._window_size == window_size:
            return
        self._aspect = aspect
        self._window_size = window_size
        
        w,h = window_size
        ah = aspect*h
        va = self._unit_vertices.copy()
        if w > ah:
            e = (w - ah)/w
            va[:,0] *= 1-e
        else:
            e = (ah - w)/ah
            va[:,1] *= 1-e
        self.set_geometry(va, None, self.triangles)

    def pad(self):
        if self._aspect is None or self._window_size is None:
            return False
        w,h = self._window_size
        return w != self._aspect * h
