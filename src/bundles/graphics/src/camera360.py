# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .camera import Camera
class CubeMapCamera(Camera):

    name = 'cube map'
    aspect = None	# Aspect ratio of projected image.  None means use full window.

    def __init__(self, cube_face_size = 1024):

        Camera.__init__(self)
        self._framebuffer = None        	# Framebuffer for rendering each face
        self._cube_face_size = cube_face_size	# Pixels
        self._view_rotations = _cube_map_face_views()   # Camera views for cube faces
        self._drawing = None	# CubeMapProjectionDrawing of that renders cube map texture
        self._vulkan_framebuffers = None
        self._vulkan_drawing = None

    def delete(self):
        fb = self._framebuffer
        if fb:
            fb.delete(make_current = True)
            self._framebuffer = None
        if self._vulkan_framebuffers:
            for vfb in self._vulkan_framebuffers:
                vfb.delete()
            self._vulkan_framebuffers = None
        if self._vulkan_drawing:
            self._vulkan_drawing.delete()
            self._vulkan_drawing = None

    def release_graphics_resources(self):
        self.delete()
        if self._drawing is not None:
            self._drawing.texture = None
            self._drawing.delete()
            self._drawing = None
            
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
        if hasattr(render, '_vulkan'):
            self._set_vulkan_face_target(view_num, render, 6)
            _adjust_light_directions(render, self._view_rotations[view_num])
            return
        fb = self._cube_face_framebuffer(render.opengl_context)
        if view_num == 0:
            render.push_framebuffer(fb)
        fb.set_cubemap_face(view_num)
        _adjust_light_directions(render, self._view_rotations[view_num])

    def combine_rendered_camera_views(self, render):
        '''Render the cube map using a projection.'''
        _adjust_light_directions(render)	# Restore light directions
        if hasattr(render, '_vulkan'):
            render.end_frame()  # Submit the final cube face.
            render.pop_framebuffer()
            self._project_vulkan_targets(render)
            return
        render.pop_framebuffer()	        # Pop the cube face framebuffer.
        cubemap = self._cube_face_framebuffer(render.opengl_context).color_texture
        proj = self._projection_drawing(render.render_size())
        proj.project_cubemap(cubemap, render) # Project cubemap to longitude/lattitude rectangle

    def _projection_drawing(self, window_size):
        d = self._drawing
        if d is None:
            vertices, triangles = self.projected_geometry()
            d = CubeMapProjectionDrawing(vertices, triangles, self.projection_direction)
            self._drawing = d
        if self.aspect is not None:
            d.set_aspect(1, window_size)
        return d

    def _cube_face_framebuffer(self, opengl_context):
        fb = self._framebuffer
        if fb is None:
            self._framebuffer = fb = _cube_map_framebuffer(opengl_context, self._cube_face_size)
        return fb

    def _vulkan_cube_face_framebuffers(self, render, count):
        framebuffers = self._vulkan_framebuffers
        size = self._cube_face_size
        bits = getattr(getattr(render, '_context', None), '_framebuffer_color_bits', 8)
        if (framebuffers is None or len(framebuffers) != count or
                any((fb.width, fb.height) != (size, size)
                    or getattr(fb, '_deleted', False)
                    or getattr(fb, '_render', render) is not render
                    or getattr(fb, 'color_bits', bits) != bits
                    for fb in framebuffers)):
            if framebuffers:
                for fb in framebuffers:
                    fb.delete()
            framebuffers = []
            try:
                for face in range(count):
                    fb = render.create_framebuffer(
                        'cube face %d' % face, size, size, alpha=False)
                    framebuffers.append(fb)
                    if not fb.activate():
                        raise RuntimeError('Could not create Vulkan cube-face framebuffer')
            except Exception:
                for fb in framebuffers:
                    fb.delete()
                self._vulkan_framebuffers = None
                raise
            self._vulkan_framebuffers = framebuffers
        return framebuffers

    def _set_vulkan_face_target(self, view_num, render, count):
        """Finish one Vulkan submission before starting the next cube face."""
        framebuffers = self._vulkan_cube_face_framebuffers(render, count)
        if view_num > 0:
            # A cube face is a complete offscreen image.  Submitting each face
            # independently prevents render/RT resources from six or twelve
            # view matrices sharing one Vulkan frame and one fence.
            render.end_frame()
            render.pop_framebuffer()
        render.push_framebuffer(framebuffers[view_num])

    def _project_vulkan_targets(self, render, layout=None):
        """Project directly from GPU face images, including 16-bit targets."""
        textures = [fb.color_texture for fb in self._vulkan_framebuffers]
        for texture in textures:
            texture.prepare_for_sampling()
        render.draw_background()
        width, height = render.render_size()
        if layout == 'top-bottom':
            split = max(1, height // 2)
            viewports = [(0, 0, width, split), (0, split, width, height - split)]
        elif layout == 'side-by-side':
            split = max(1, width // 2)
            viewports = [(0, 0, split, height), (split, 0, width - split, height)]
        else:
            viewports = [(0, 0, width, height)]
        from numpy import array, float32
        try:
            for eye, (x, y, w, h) in enumerate(viewports):
                if w <= 0 or h <= 0:
                    continue
                render.set_viewport(x, y, w, h)
                parameters = array([
                    float(isinstance(self, DomeCamera)), max(1, w / h),
                    max(1, h / w), 0, *render._background_color], dtype=float32)
                render.draw_fullscreen_quad_textures(
                    [t._handle for t in textures[6*eye:6*eye+6]],
                    'cube_project', 'main', 'main', parameters, has_depth=True)
        finally:
            render.full_viewport()

    def _draw_vulkan_projection(self, rgba, render):
        from .drawing import rgba_drawing, draw_overlays
        drawing = self._vulkan_drawing
        if drawing is None:
            from . import Drawing
            self._vulkan_drawing = drawing = Drawing('Vulkan cube projection')
        rgba_drawing(drawing, rgba, (-1, -1), (2, 2))
        drawing.opaque_texture = True
        draw_overlays([drawing], render)

    def view_width(self, point):
        return view_width_360(point, self.position.origin())

    def projected_geometry(self):
        '''
        Return vertices and triangles that cube map projects onto.
        Vertex x,y coordinates range from -1 to 1 spanning full window.
        Derived class must define this function.
        '''
        raise NotImplementedError('CubeMapCamera %s did not implement projected_geometry() method.')
        
    def projection_direction(self, x, y):
        '''
        Cube map direction vector for vertex position (x,y) (range -1 to 1)
        -z axis is at x,y = 0,0.
        Derived class must define this function.
        '''
        raise NotImplementedError('CubeMapCamera %s did not implement projection_direction() method.')

class Mono360Camera(CubeMapCamera):
    name = 'mono 360'

    def __init__(self, cube_face_size = 1024, projection_size = (360, 180)):
        CubeMapCamera.__init__(self, cube_face_size = cube_face_size)

        self._projection_size = projection_size

    def projected_geometry(self):
        '''Return vertices and triangles.'''
        return _rectangle_geometry(self._projection_size)

    def projection_direction(self, x, y):
        '''
        Equirectangular projection maps image x to longitude and image y to lattitude.
        Direction vector for texture position (x,y) (range -1 to 1)
        -z axis is at x,y = 0,0.
        '''
        return _equirectangular_direction(x,y)

def _equirectangular_direction(x, y):
    from math import pi, cos, sin
    theta, phi = x * pi, (1-y) * (pi/2)
    ct, st = cos(theta), sin(theta)
    cp, sp = cos(phi), sin(phi)
    return (st*sp,cp,-ct*sp)


def _project_vulkan_cube_faces(faces, width, height, camera,
                               background=(0, 0, 0, 1)):
    """Project six ordinary Vulkan face targets into a display image."""
    import numpy as np
    width, height = max(1, int(width)), max(1, int(height))
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    valid = np.ones((height, width), dtype=bool)

    if isinstance(camera, DomeCamera):
        lx = np.broadcast_to(
            x * (width / height if width > height else 1), (height, width))
        ly = np.broadcast_to(
            y * (height / width if height > width else 1), (height, width))
        radius = np.sqrt(lx * lx + ly * ly)
        valid = radius <= 1.0
        theta = np.arctan2(ly, lx)
        phi = 0.5 * np.pi * np.minimum(radius, 1.0)
        sin_phi = np.sin(phi)
        dx = np.cos(theta) * sin_phi
        dy = np.sin(theta) * sin_phi
        dz = -np.cos(phi)
    else:
        xx = np.broadcast_to(x, (height, width))
        yy = np.broadcast_to(y, (height, width))
        theta = xx * np.pi
        phi = (1.0 - yy) * (0.5 * np.pi)
        sin_phi = np.sin(phi)
        dx = np.sin(theta) * sin_phi
        dy = np.cos(phi)
        dz = -np.cos(theta) * sin_phi

    rgba8 = np.array(
        [max(0, min(255, round(255 * c))) for c in background],
        dtype=np.uint8)
    result = np.empty((height, width, 4), dtype=np.uint8)
    result[:] = rgba8

    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    face_index = np.zeros((height, width), dtype=np.int8)
    sc = np.zeros((height, width), dtype=np.float32)
    tc = np.zeros((height, width), dtype=np.float32)

    mx = (ax >= ay) & (ax >= az)
    mask = mx & (dx >= 0)
    face_index[mask] = 0
    sc[mask], tc[mask] = -dz[mask] / ax[mask], -dy[mask] / ax[mask]
    mask = mx & (dx < 0)
    face_index[mask] = 1
    sc[mask], tc[mask] = dz[mask] / ax[mask], -dy[mask] / ax[mask]
    my = (~mx) & (ay >= az)
    mask = my & (dy >= 0)
    face_index[mask] = 2
    sc[mask], tc[mask] = dx[mask] / ay[mask], dz[mask] / ay[mask]
    mask = my & (dy < 0)
    face_index[mask] = 3
    sc[mask], tc[mask] = dx[mask] / ay[mask], -dz[mask] / ay[mask]
    mz = ~(mx | my)
    mask = mz & (dz >= 0)
    face_index[mask] = 4
    sc[mask], tc[mask] = dx[mask] / az[mask], -dy[mask] / az[mask]
    mask = mz & (dz < 0)
    face_index[mask] = 5
    sc[mask], tc[mask] = -dx[mask] / az[mask], -dy[mask] / az[mask]

    for face_number, face in enumerate(faces):
        if face is None:
            continue
        mask = valid & (face_index == face_number)
        if not mask.any():
            continue
        fh, fw = face.shape[:2]
        iu = np.clip(np.rint((sc[mask] + 1) * 0.5 * (fw - 1)),
                     0, fw - 1).astype(np.intp)
        iv = np.clip(np.rint((tc[mask] + 1) * 0.5 * (fh - 1)),
                     0, fh - 1).astype(np.intp)
        result[mask] = face[iv, iu]
    return result
    
class DomeCamera(CubeMapCamera):
    name = 'dome'
    aspect = 1
    
    def __init__(self, cube_face_size = 1024, grid_spacing = 1):
        CubeMapCamera.__init__(self, cube_face_size = cube_face_size)

        self._grid_spacing = grid_spacing	# Degrees

    def projected_geometry(self):
        '''Return vertices and triangles.'''
        return _circle_geometry(self._grid_spacing)
        
    def projection_direction(self, x, y):
        '''
        Fisheye projection maps image x,y to hemisphere point where 90*sqrt(x*x+y*y) is angle from pole.
        Direction vector for texture position (x,y) (range -1 to 1)
        -z axis is at x,y = 0,0.
        '''
        from math import atan2, pi, sqrt, cos, sin
        theta = atan2(y, x)
        phi = 0.5*pi*sqrt(x*x + y*y)
        ct, st = cos(theta), sin(theta)
        cp, sp = cos(phi), sin(phi)
        return (ct*sp, st*sp, -cp)

    
class Stereo360Camera(CubeMapCamera):

    name = 'stereo 360'

    def __init__(self, layout = 'top-bottom', cube_face_size = 1024):

        Camera.__init__(self)
        self.eye_separation_scene = 0.2			# Angstroms
        self._framebuffer = {'left':None, 'right':None} # Framebuffer for rendering each face
        self._cube_face_size = cube_face_size		# Pixels
        self._projection_size = (360,180)		# Grid size for projecting cubemap.
        self._drawing = {'left':None, 'right':None}	# Drawing of rectangle with cube map texture
        self._vulkan_framebuffers = None
        self._vulkan_drawing = None
        v = _cube_map_face_views()
        self._view_rotations = v + v		# Camera views for cube faces
        self.layout = layout			# Packing of left/right eye images: top-bottom or side-by-side

    def delete(self):
        for fb in self._framebuffer.values():
            if fb:
                fb.delete(make_current = True)
        self._framebuffer = {}

        for d in self._drawing.values():
            if d is not None:
                d.texture = None
                d.delete()
        self._drawing = {}
        if self._vulkan_framebuffers:
            for fb in self._vulkan_framebuffers:
                fb.delete()
            self._vulkan_framebuffers = None
        if self._vulkan_drawing:
            self._vulkan_drawing.delete()
            self._vulkan_drawing = None

    def release_graphics_resources(self):
        self.delete()
        self._framebuffer = {'left': None, 'right': None}
        self._drawing = {'left': None, 'right': None}
        
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
        if hasattr(render, '_vulkan'):
            self._set_vulkan_face_target(view_num, render, 12)
            if view_num in (0, 6):
                eye = 'left' if view_num == 0 else 'right'
                self._set_stereo_360_shader_parameters(render, eye)
            _adjust_light_directions(render, self._view_rotations[view_num])
            return
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
        if hasattr(render, '_vulkan'):
            render.end_frame()
            render.pop_framebuffer()
            self._project_vulkan_targets(render, self.layout)
            return
        render.pop_framebuffer()	        # Pop the cube face framebuffer.
        for eye in ('left', 'right'):
            cubemap = self._cube_face_framebuffer(eye, render.opengl_context).color_texture
            proj = self._projection_drawing(eye)
            proj.project_cubemap(cubemap, render) # Project cubemap to longitude/lattitude rectangle

    def _cube_face_framebuffer(self, eye, opengl_context):
        fb = self._framebuffer[eye]
        if fb is None:
            self._framebuffer[eye] = fb = _cube_map_framebuffer(opengl_context, self._cube_face_size)
        return fb

    def _projection_drawing(self, eye):
        d = self._drawing[eye]
        if d is None:
            vertices, triangles = _rectangle_geometry(self._projection_size)
            d = CubeMapProjectionDrawing(vertices, triangles, _equirectangular_direction)
            self._drawing[eye] = d
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
    from chimerax.geometry import vector
    return 2 * pi * vector.distance(origin, point)

def view_all_360(bounds, cam_position):
    center, size = bounds.center(), bounds.width()
    shift = center - cam_position.origin() + 2*size*cam_position.z_axis()
    from chimerax.geometry import translation
    return translation(shift) * cam_position

# Camera rotations for 6 cube faces. Face order +x,-x,+y,-y,+z,-z.
def _cube_map_face_views():
    from chimerax.geometry import Place
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

def _rectangle_geometry(size = (360, 180)):
    # Compute vertices (-1 to 1 range) for rectangular grid.
    from numpy import arange, empty, float32, int32
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

    # Compute triangles for rectangular grid
    ta = empty((h-1,w-1,2,3), int32)
    for j in range(h-1):
        for i in range(w-1):
            ta[j,i,0,:] = (j*w+i, j*w+(i+1), (j+1)*w+(i+1))
            ta[j,i,1,:] = (j*w+i, (j+1)*w+(i+1), (j+1)*w+i)
    ta = ta.reshape(((h-1)*(w-1)*2, 3))

    return va,ta

def _circle_geometry(spacing = 1):
    '''spacing is in degrees at perimeter'''
    vertices = []
    triangles = []
    from math import pi, cos, sin
    step = 2*pi * spacing/360
    nr = max(2, int(1/step))
    prev_ring = None
    for r in range(nr):
        rad = r / (nr-1)
        nc = max(1, int(2*pi*rad / step))
        ring = []
        for c in range(nc):
            a = (c/nc) * 2*pi
            ring.append((rad*cos(a), rad*sin(a), 0))
        offset = len(vertices)
        vertices.extend(ring)
        if prev_ring is not None:
            triangles.extend(_stitched_triangles(prev_ring, prev_offset, ring, offset))
        prev_ring = ring
        prev_offset = offset

    from numpy import array, float32, int32
    va = array(vertices, float32)
    ta = array(triangles, int32)

    return va, ta

def _stitched_triangles(loop1_vertices, offset1, loop2_vertices, offset2):
    triangles = []
    n1, n2 = len(loop1_vertices), len(loop2_vertices)
    i1 = i2 = 0
    from chimerax.geometry import distance
    while i1 < n1 or i2 < n2:
        v1, v2 = loop1_vertices[i1%n1], loop2_vertices[i2%n2]
        vn1, vn2 = loop1_vertices[(i1+1)%n1], loop2_vertices[(i2+1)%n2]
        if i2 == n2:
            step1 = True
        elif i1 == n1:
            step1 = False
        else:
            step1 = (distance(v2,vn1) < distance(v1,vn2))
        i3 = (offset1 + (i1+1)%n1) if step1 else (offset2 + (i2+1)%n2)
        triangles.append((offset1+i1%n1, offset2+i2%n2, i3))
        if step1:
            i1 += 1
        else:
            i2 += 1

    return triangles
    
from . import Drawing
class CubeMapProjectionDrawing(Drawing):
    def __init__(self, vertices, triangles, direction_function):
        Drawing.__init__(self, 'cube map projection')

        tc = self._texture_coordinate_array(vertices, direction_function)
        
        # Create rectangle drawing with sphere point texture coordinates.
        self.set_geometry(vertices, None, triangles)
        self.color = (255,255,255,255)
        self.use_lighting = False
        self.texture_coordinates = tc
        self.opaque_texture = True

        self._scale = (1,1)

    def _texture_coordinate_array(self, vertices, direction_function):
        # Compute direction vectors as texture coordinates
        from numpy import array, float32
        tc = array([direction_function(vx, vy) for vx,vy,vz in vertices], float32)
        return tc

    def set_aspect(self, aspect, window_size):
        w,h = window_size
        ah = aspect*h
        if w > ah:
            e = (w - ah)/w
            s = (1-e, 1)
        else:
            e = (ah - w)/ah
            s = (1, 1-e)
        self._scale = s

    def project_cubemap(self, cubemap_texture, render):
        '''Project cubemap onto overlay drawing'''
        if self._scale != (1,1):
            render.draw_background()
        dc = render.disable_capabilities
        render.disable_capabilities |= render.SHADER_STEREO_360
        self.texture = cubemap_texture
        from .drawing import draw_overlays
        draw_overlays([self], render, scale = self._scale)
        render.disable_capabilities = dc
