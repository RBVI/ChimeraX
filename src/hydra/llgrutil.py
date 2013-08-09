import llgr
from numpy import array, float32

use_llgr = True
#use_llgr = False

def initialize_llgr():
    llgr.set_output('pyopengl')
    llgr.clear_all()
    set_shaders()

_program_id = None
def set_shaders():
    global _program_id
    _program_id = llgr.next_program_id()
    with open("../shaders/vertexShader150.txt") as f:
        vertex_shader = f.read()
    with open("../shaders/fragmentShader150.txt") as f:
        fragment_shader = f.read()
    with open("../shaders/vertexPickShader150.txt") as f:
        pick_vertex_shader = f.read()
    llgr.create_program(_program_id, vertex_shader, fragment_shader,
                        pick_vertex_shader)

def set_light_parameters(viewer):
    v = viewer
    specular = array(tuple(v.key_light_specular_color) + (1,), dtype=float32)
    ambient = array(tuple(v.ambient_light_color) + (1,), dtype=float32)
    fill_diffuse = array(tuple(v.fill_light_diffuse_color) + (1,), dtype=float32)
    fill_position = array(tuple(v.fill_light_position) + (0,), dtype=float32)
    key_diffuse = array(tuple(v.key_light_diffuse_color) + (1,), dtype=float32)
    key_position = array(tuple(v.key_light_position) + (0,), dtype=float32)
    shininess = array([v.key_light_specular_exponent], dtype=float32)
    llgr.set_uniform(0, 'Ambient', llgr.FVec4, ambient)
    llgr.set_uniform(0, 'KeyDiffuse', llgr.FVec4, key_diffuse)
    llgr.set_uniform(0, 'KeySpecular', llgr.FVec4, specular)
    llgr.set_uniform(0, 'KeyPosition', llgr.FVec4, key_position)
    llgr.set_uniform(0, 'Shininess', llgr.FVec1, shininess)
    llgr.set_uniform(0, 'FillDiffuse', llgr.FVec4, fill_diffuse)
    llgr.set_uniform(0, 'FillPosition', llgr.FVec4, fill_position)

def set_background_color(rgba):
    r,g,b,a = rgba
    llgr.set_clear_color(r,g,b,a)

def set_projection_matrix(m44):
    p44 = array(m44, float32)
    llgr.set_uniform_matrix(_program_id, 'ProjectionMatrix', False, llgr.Mat4x4, p44)

def set_modelview_matrix(m44):
    mv44 = array(m44, float32)
    llgr.set_uniform_matrix(_program_id, 'ModelViewMatrix', False, llgr.Mat4x4, mv44)
    nm33 = mv44[:3,:3].copy()
    llgr.set_uniform_matrix(_program_id, 'NormalMatrix', False, llgr.Mat3x3, nm33)

class llgr_sphere:
    def __init__(self, radius, center, color):

        color_id = llgr.next_data_id()
        assert len(color) == 4
        rgba = array(color, dtype=float32)
        llgr.create_singleton(color_id, rgba)

        self.matrix_id = matrix_id = llgr.next_matrix_id()
        cx, cy, cz = center
        mat = array(((1,0,0,0),
                     (0,1,0,0),
                     (0,0,1,0),
                     (cx,cy,cz,1)), dtype=float32)
        llgr.create_matrix(matrix_id, mat, False)

        self.object_id = obj_id = llgr.next_object_id()
        ai = llgr.AttributeInfo("color", color_id, 0, 0, 4, llgr.Float)
        llgr.add_sphere(obj_id, radius, _program_id, matrix_id, [ai])

    def delete(self):
        llgr.delete_object(self.object_id)
        self.object_id = None
        llgr.delete_matrix(self.matrix_id)
        self.matrix_id = None

class llgr_molecule:
    def __init__(self, xyz, radii, rgba):
        self.spheres = [llgr_sphere(radii[a], xyz[a], rgba[a]/255.0) for a in range(len(xyz))]
    def delete(self):
        [s.delete() for s in self.spheres]
        self.spheres = []

class llgr_surface:
    def __init__(self, varray, narray, tarray, color, carray):

        self.buffer_ids = buf_ids = []

        vn_id = llgr.next_data_id()
        from numpy import float32, concatenate, array
        vn = concatenate([varray, narray])
        llgr.create_buffer(vn_id, llgr.ARRAY, vn)
        buf_ids.append(vn_id)

        tri_id = llgr.next_data_id()
        llgr.create_buffer(tri_id, llgr.ELEMENT_ARRAY, tarray)
        buf_ids.append(tri_id)

        color_id = llgr.next_data_id()
        if carray is None:
            rgba = array(color, dtype=float32)
            llgr.create_singleton(color_id, rgba)
        else:
            rgba = llgr.create_buffer(color_id, llgr.ARRAY, carray/255.0)
            buf_ids.append(color_id)

        uniform_scale_id = llgr.next_data_id()
        llgr.create_singleton(uniform_scale_id, array([1, 1, 1], dtype=float32))
        
        matrix_id = 0		# default identity matrix

        obj_id = llgr.next_object_id()
        AI = llgr.AttributeInfo
        ais = [
            AI("color", color_id, 0, 0, 4, llgr.Float),
            AI("position", vn_id, 0, 0, 3, llgr.Float),
            AI("normal", vn_id, varray.nbytes, 0, 3, llgr.Float),
            AI("instanceScale", uniform_scale_id, 0, 0, 3, llgr.Float),
            ]

        global _program_id
        llgr.create_object(obj_id, _program_id, matrix_id, ais,
                           llgr.Triangles, 0, tarray.size, tri_id, llgr.UInt)
        self.object_id = obj_id

    def delete(self):

        llgr.delete_object(self.object_id)
        for id in self.buffer_ids:
            llgr.delete_buffer(id)
        self.object_id = None
        self.buffer_ids = []
        
def update_llgr_graphics(surface):
    for p in surface.plist:
        update_llgr_surface_piece(p)

def update_llgr_surface_piece(p):
    if p.triangles is None or hasattr(p, 'llgr_surface'):
      return
    from . import gui
    if p.shift_and_scale is None:
      s = llgr_surface(p.vertices, p.normals, p.triangles, p.color_rgba, p.vertex_colors)
      gui.show_info('Created llgr surface, %d triangles' % len(p.triangles))
    else:
      # Assume geometry is a sphere
      xyz = p.shift_and_scale[:,:3]
      radii = p.shift_and_scale[:,3]
      rgba = p.instance_colors
      s = llgr_molecule(xyz, radii, rgba)
      gui.show_info('created llgr molecule, %d atoms' % (len(xyz),))
    p.llgr_surface = s

    # Make deleting surface piece delete llgr surface
    def delete(self=p):
        self.delete_original()
        if use_llgr and hasattr(self, 'llgr_surface'):
            self.llgr_surface.delete()
    p.delete_original = p.delete
    p.delete = delete

def render(viewer):
    v = viewer
    w,h = v.window_size
    from OpenGL import GL
    GL.glViewport(0,0,w,h)
    set_background_color(v.background_rgba)
    set_light_parameters(v)
    set_projection_matrix(v.projection_matrix())
    if v.models:
        set_modelview_matrix(v.model_view_matrix(v.models[0]))
        for m in v.models:
            if hasattr(m, 'update_graphics'):
                m.update_graphics(v)
            update_llgr_graphics(m)
    llgr.render()
#    from . import gui
#    gui.show_info('rendered %d models' % len(v.models))
