from OpenGL import GL

class Shaders:

    def __init__(self, lighting_params):
                
        self.shader_programs = {}
        self.current_shader_program = None

        self.current_projection_matrix = None   # Used when switching shaders
        self.current_model_view_matrix = None   # Used when switching shaders
        self.current_model_matrix = None        # Used for optimizing model view matrix updates
        self.current_inv_view_matrix = None        # Used for optimizing model view matrix updates

        self.lighting_params = lighting_params

    def use_shader(self, **kw):

        default_capabilities = ('USE_LIGHTING',)
        capabilities = set(default_capabilities)

        options = {'lighting':'USE_LIGHTING', 'texture2d':'USE_TEXTURE_2D',
                   'shiftAndScale':'USE_INSTANCING_SS', 'instancing':'USE_INSTANCING_44',
                   'selected':'USE_HATCHING'}
#                   'unselected':'USE_DIMMING'}
        for opt,onoff in kw.items():
            if opt in options:
                cap = options[opt]
                if onoff:
                    capabilities.add(cap)
                else:
                    capabilities.discard(cap)

        p = self.opengl_shader(capabilities)

        if p != self.current_shader_program:
#            print ('changed shader',
#                   self.current_shader_program.capabilities if self.current_shader_program else None, p.capabilities)
            self.current_shader_program = p
            GL.glUseProgram(p.program_id)
            if 'USE_LIGHTING' in capabilities:
                self.set_shader_lighting_parameters()
            self.set_projection_matrix()
            self.set_model_view_matrix()
            if 'USE_TEXTURE_2D' in capabilities:
                GL.glUniform1i(p.uniform_id("tex2d"), 0)    # Texture unit 0.

        return p

    def opengl_shader(self, capabilities = ('USE_LIGHTING',), glsl_version = '150'):
        
        ckey = tuple(sorted(capabilities))
        p = self.shader_programs.get(ckey)
        if not p is None:
            return p

        p = Shader(capabilities, glsl_version)
        self.shader_programs[ckey] = p

        return p
        
    def set_projection_matrix(self, pm = None):

        if pm is None:
            if self.current_projection_matrix is None:
                return
            pm = self.current_projection_matrix
        else:
            self.current_projection_matrix = pm
        p = self.current_shader_program
        if not p is None:
            GL.glUniformMatrix4fv(p.uniform_id('projection_matrix'), 1, True, pm)

    def set_model_view_matrix(self, view_matrix_inverse = None, model_matrix = None, matrix = None):

        if not matrix is None:
            mv4 = matrix
            self.current_model_view_matrix = mv4
            self.current_model_matrix = None
            self.current_inv_view_matrix = None
        elif model_matrix is None:
            mv4 = self.current_model_view_matrix
            if mv4 is None:
                return
        else:
            if model_matrix == self.current_model_matrix:
                from numpy import all
                if all(view_matrix_inverse == self.current_inv_view_matrix):
                    return
            from . import matrix
#            v = matrix.invert_matrix(view_matrix)      # Slow, 10000 per second.
            v = view_matrix_inverse
            m = model_matrix
            # TODO: optimize matrix multiply.  Rendering bottleneck with 200 models open.
            mv = matrix.multiply_matrices(v, m)
            mv4 = opengl_matrix(mv)
            self.current_model_view_matrix = mv4
            self.current_model_matrix = m
            self.current_inv_view_matrix = v

        p = self.current_shader_program
        if not p is None:
            var_id = p.uniform_id('model_view_matrix')
            # Note: Getting about 5000 glUniformMatrix4fv() calls per second on 2013 Mac hardware.
            # This can be a rendering bottleneck for large numbers of models or instances.
            GL.glUniformMatrix4fv(var_id, 1, False, mv4)

    def set_shader_lighting_parameters(self):

        p = self.current_shader_program.program_id
        lp = self.lighting_params

        # Key light
        key_light_pos = GL.glGetUniformLocation(p, b"key_light_position")
        GL.glUniform3f(key_light_pos, *lp.key_light_position)
        key_diffuse = GL.glGetUniformLocation(p, b"key_light_diffuse_color")
        GL.glUniform3f(key_diffuse, *lp.key_light_diffuse_color)

        # Key light specular
        key_specular = GL.glGetUniformLocation(p, b"key_light_specular_color")
        GL.glUniform3f(key_specular, *lp.key_light_specular_color)
        key_shininess = GL.glGetUniformLocation(p, b"key_light_specular_exponent")
        GL.glUniform1f(key_shininess, lp.key_light_specular_exponent)

        # Fill light
        fill_light_pos = GL.glGetUniformLocation(p, b"fill_light_position")
        GL.glUniform3f(fill_light_pos, *lp.fill_light_position)
        fill_diffuse = GL.glGetUniformLocation(p, b"fill_light_diffuse_color")
        GL.glUniform3f(fill_diffuse, *lp.key_light_diffuse_color)

        # Ambient light
        ambient = GL.glGetUniformLocation(p, b"ambient_color")
        GL.glUniform3f(ambient, *lp.ambient_light_color)

class Shader:

    def __init__(self, capabilities, glsl_version = '150'):

        self.capabilities = capabilities
        self.program_id = self.compile_shader(capabilities, glsl_version)
        self.uniform_ids = {}
        self.attribute_ids = {}
        
    def uniform_id(self, name):
        uids = self.uniform_ids
        if name in uids:
            uid = uids[name]
        else:
            p = self.program_id
            uids[name] = uid = GL.glGetUniformLocation(p, name.encode('utf-8'))
        return uid

    def attribute_id(self, name):
        aids = self.attribute_ids
        if name in aids:
            aid = aids[name]
        else:
            p = self.program_id
            aids[name] = aid = GL.glGetAttribLocation(p, name.encode('utf-8'))
        return aid

    def compile_shader(self, capabilities, glsl_version = '150'):

        from os.path import dirname, join
        d = dirname(__file__)
        f = open(join(d,'vshader%s.txt' % glsl_version), 'r')
        vshader = insert_define_macros(f.read(), capabilities)
        f.close()

        f = open(join(d,'fshader%s.txt' % glsl_version), 'r')
        fshader = insert_define_macros(f.read(), capabilities)
        f.close()

        from OpenGL.GL import shaders
        vs = shaders.compileShader(vshader, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fshader, GL.GL_FRAGMENT_SHADER)

        prog_id = shaders.compileProgram(vs, fs)
        return prog_id


# Add #define lines after #version line of shader
def insert_define_macros(shader, capabilities):
    defs = '\n'.join('#define %s 1' % c for c in capabilities)
    v = shader.find('#version')
    eol = shader[v:].find('\n')+1
    s = shader[:eol] + defs + '\n' + shader[eol:]
    return s
    
def opengl_matrix(m):
    return ((m[0][0], m[1][0], m[2][0], 0),
            (m[0][1], m[1][1], m[2][1], 0),
            (m[0][2], m[1][2], m[2][2], 0),
            (m[0][3], m[1][3], m[2][3], 1))
