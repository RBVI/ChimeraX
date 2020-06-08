#
# This is experimental code to explore the possibility rendering graphics in a separate thread
# by making a list of opengl calls that are repeated in a C++ thread.  Some of the calls will have
# there arguments modified as the camera view changes.  The goal of this is to maintain high rendering
# rates needed by VR headsets when calculations taking more than 1/90 second are running in the main thread.
#
# Wrap PyOpenGL to add calls to a list instead of executing them, then execute them later when requested.
# This allows the call list to be executed multiple times.
#
# For OpenGL calls that return a value like glGenBuffers() which is recorded by Python code this list
# mechanism won't work.  Raise an error if an attempt is made to call those routines.
#
def wrap_opengl_routines(dict):
    # Cannot put calls returning (non-error code) values in call lists since there is no mechanism to use
    # the return value.
    opengl_calls_returning_values = set(('glAttachShader', 'glCheckFramebufferStatus', 'glCreateProgram'
                                         'glGenBuffers', 'glGenFramebuffers', 'glGenRenderbuffers',
                                         'glGenTextures', 'glGenVertexArrays',
                                         'glGetDebugMessageLog', 'glGetError', 'glGetInteger', 'glGetIntegerv',
                                         'glGetProgramiv', 'glGetString', 'glGetUniformBlockIndex',
                                         'glGetUniformLocation'))
    from OpenGL import GL
    for name, value in vars(GL).items():
        if name.startswith('gl'):
            if name in opengl_calls_returning_values:
                dict[name] = lambda *args, name=name: raise_call_list_error(name)
            else:
                dict[name] = AddGLCall(name, value)
        else:
            dict[name] = value

def raise_call_list_error(name):
    raise RuntimeError('Cannot put OpenGL call %s in call list since it returns a value' % name)

_call_list = None
def start_gl_call_list():
    global _call_list
    _call_list = GLCallList()
    return _call_list

def call_list():
    return _call_list

class AddGLCall:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def __call__(self, *args, **kw):
        c = GLCall(self.name, self.function, args, kw)
        global _call_list
        _call_list.add_call(c)
        c._calltrace = self._traceback()

    def _traceback(self, depth = 4, up = 2):
        import inspect
        f = inspect.currentframe()
        level = 0
        from os.path import basename
        lines = []
        while f is not None:
            if level >= up + depth:
                break
            if level >= up:
                lines.append('%s %d' % (basename(f.f_code.co_filename), f.f_lineno))
            f = f.f_back
            level += 1
        s =  ', '.join(lines)
        return s

wrap_opengl_routines(globals())

class GLCall:
    def __init__(self, name, function, args, keywords):
        self.name = name
        self.function = function
        self.args = args
        self.keywords = keywords
        self._calltrace = ''
        self._has_computed_arguments = None
    def __str__(self):
        return c.name
    def report(self, trace=False, shaders={}, shader_id=None, vao_names={}, fb_names={}):
        n = self.name
        if n == 'glUseProgram' and self.args[0] in shaders:
            f = '%s(%s)' % (n, str(shaders[self.args[0]]))
        elif n.startswith('glUniform') and shader_id is not None and shader_id in shaders:
            s = shaders[shader_id]
            unames = {uname:uid for uid,uname in s.uniform_ids.items()}
            uname = unames.get(self.args[0], 'unknown')
            f = '%s(%s,...)' % (n, uname)
        elif n == 'glBindVertexArray' and self.args[0] in vao_names:
            f = '%s(%s)' % (n, vao_names[self.args[0]])
        elif n == 'glBindFramebuffer':
            fb_id = self.args[1]
            fb_name = fb_names.get(fb_id, str(fb_id))
            f = '%s(%s)' % (n, fb_name)
        else:
            f = n
        return ('%s %s' % (f, self._calltrace)) if trace else f
    def new_shader_id(self):
        return self.args[0] if self.name == 'glUseProgram' else None
    def __call__(self):
        args = self.compute_arguments()
        if False and self._has_computed_arguments:
            import sys
            sys.__stderr__.write('Calling %s %s, args %s, precompute %s\n'
                                 % (self.name, self._calltrace, str(args), str(self.args)))
        try:
            result = self.function(*args, **self.keywords)
        except Exception:
            arg_types = ', '.join(str(type(a)) for a in self.args)
            print('Error executing opengl call %s %s, arg types %s' % (self._calltrace, self.name, arg_types))
            raise
        return result
    def compute_arguments(self):
        hca = self._has_computed_arguments
        if hca is None:
            self._has_computed_arguments = hca = self._check_for_computed_args()
        if hca:
            args = tuple((a() if callable(a) else a) for a in self.args)
        else:
            args = self.args
        return args
    def _check_for_computed_args(self):
        for a in self.args:
            if callable(a):
                return True
        return False
        
    def replace_argument(self, n, new_arg):
        self.args = tuple((new_arg if i == n else a) for i,a in enumerate(self.args))
#        print ('replaced', self._calltrace, self.name, 'argument', n, 'with', new_arg)

class GLCallList:
    def __init__(self):
        self._gl_calls = []
    @property
    def calls(self):
        return self._gl_calls
    def add_call(self, gl_call):
        self._gl_calls.append(gl_call)
    def __call__(self):
        for gl_call in self._gl_calls:
            gl_call()
    def __len__(self):
        return len(self._gl_calls)
    def __bool__(self):
        return True
    def last_call(self):
        nc = len(self)
        return None if nc == 0 else self._gl_calls[nc-1]
    def __str__(self):
        return self.report()
    def report(self, trace=False, shaders={}, vao_names={}, fb_names={}):
        lines = []
        kw = {'trace':trace, 'shaders':shaders, 'shader_id':-1, 'vao_names':vao_names, 'fb_names':fb_names}
        for c in self._gl_calls:
            lines.append(c.report(**kw))
            sid = c.new_shader_id()
            if sid is not None:
                kw['shader_id'] = sid
        return '\n'.join(lines)
    def replace_argument(self, calls, n, new_arg):
        for c in calls:
            c.replace_argument(n, new_arg)

# Calculate projection matrix based on new camera position for updating recorded opengl calls.
class ProjectionCalc:
    def __init__(self, view, view_num, window_size):
        self.view = view
        self.view_num = view_num
        self.window_size = window_size
        self._near_far = None
    def near_far(self):
        if self._near_far is None:
            v = self.view
            self._near_far = v.near_far_distances(v.camera, self.view_num)
        return self._near_far
    def projection_matrix(self):
        c = self.view.camera
        pm = c.projection_matrix(self.near_far(), self.view_num, self.window_size)
        self._near_far = None
        return pm

class MatrixFunc:
    '''
    Function that evaluates to a matrix.
    Allows multiplying and inverse operations which return a new MatrixFunc.
    '''
    def __init__(self, name, matrix_func):
        self._name = name
        self._matrix_func = matrix_func
    def __str__(self):
        return '%s matrix func' % self._name
    def __call__(self):
        return self._matrix_func()
    def inverse(self):
        return MatrixFunc(str(self)+' inverse', lambda: self().inverse())
    def __mul__(self, place):
        if isinstance(place, MatrixFunc):
            mf = MatrixFunc(str(self)+' times '+str(place), lambda: self() * place())
        else:
            mf = MatrixFunc(str(self)+' times place', lambda: self() * place)
        return mf
    def opengl_matrix(self):
        return MatrixFunc(str(self)+' opengl matrix', lambda: self().opengl_matrix())

class Mat44Func:
    def __init__(self, name, mat44_func, nmat):
        self._name = name
        self._mat44_func = mat44_func
        self._nmat = nmat
    def __str__(self):
        return '%s matrices func' % self._name
    def __call__(self):
        return self._mat44_func()
    def __len__(self):
        return self._nmat
    @property
    def nbytes(self):
        return 64 * self._nmat
        

# Set view matrix used in setting model view shader variable.
class ViewMatrixFunc(MatrixFunc):
    def __init__(self, view, view_num):
        self._view = view
        self._view_num = view_num
        MatrixFunc.__init__(self, 'view matrix', self._view_matrix)
    def _view_matrix(self):
        return self._view.camera.get_position(self._view_num)
    
class ShadowMatrixFunc:
    def __init__(self, render, light_direction, center, radius, depth_bias):
        self._render = render
        self._light_direction = light_direction
        self._center = center
        self._radius = radius
        self._depth_bias = depth_bias
    @property
    def lvinv(self):
        def lvfunc():
            lvinv, stf = self._render.shadow_transforms(self._light_direction(),
                                                        self._center, self._radius, self._depth_bias)
            return lvinv
        return MatrixFunc('shadow light matrix', lvfunc)
    @property
    def stf(self):
        def stffunc():
            lvinv, stf = self._render.shadow_transforms(self._light_direction(),
                                                        self._center, self._radius, self._depth_bias)
            return stf
        return MatrixFunc('shadow transform', stffunc)

def replay_opengl(view, drawings, camera, swap_buffers):
    if view._cam_only_change and drawings is None:
        opengl_calls = view._opengl_calls
        #print ('cam only change')
        if opengl_calls is None:
            r = view.render
            view._opengl_calls = r.record_opengl_calls(True)
            r._opengl_context.current_shader_program = None  # TODO: Hack to force setting shader program
        else:
            _replay_opengl_calls(opengl_calls, view, camera, swap_buffers)
            return True
    else:
        print ('not cam only change')
        view._opengl_calls = None
    return False


def call_opengl_list(view, trace=False):
    opengl_calls = getattr(view, '_opengl_calls', None)
    if opengl_calls:
        r = view.render
        r.record_opengl_calls(False)
        opengl_calls()
        shaders = {p.program_id:p for p in r._opengl_context.shader_programs.values()}
        vao_names = {b._vao_id:b._name for b in r.opengl_context._bindings}
        fb_names = {fb._fbo:fb.name for fb in r.opengl_context._framebuffers}
        np = len(set(c.args[0] for c in opengl_calls.calls if c.name == 'glUseProgram' and c.args[0] != 0))
        print('called opengl list first time', len(opengl_calls), 'calls, ', np, 'shader programs')
        print(opengl_calls.report(trace=trace, shaders=shaders, vao_names=vao_names, fb_names=fb_names))


def _replay_opengl_calls(opengl_calls, view, camera, swap_buffers):
    opengl_calls()
    if swap_buffers:
        r = view.render
        c = view.camera if camera is None else camera
        if c.do_swap_buffers():
            r.swap_buffers()
        view.redraw_needed = False
        r.done_current()

    #print('called opengl list again', len(opengl_calls))
