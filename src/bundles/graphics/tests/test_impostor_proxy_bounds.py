'''
Check that impostor quads computed by the vertex shader cover the projected
sphere or cylinder surface beyond the near plane.  Uses transform feedback
to capture the quad corners, so no image comparison is needed.

Requires a desktop OpenGL context, otherwise the tests are skipped.
'''

import ctypes
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def gl_context():
    from Qt.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
    from Qt.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        pytest.skip('OpenGL context unavailable')
    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        surface.destroy()
        pytest.skip('Offscreen OpenGL surface unavailable')
    yield context
    context.doneCurrent()
    surface.destroy()
    assert app is not None


def _perspective(near=1.0, far=100.0, field_of_view=30):
    f = 1 / np.tan(np.radians(field_of_view) / 2)
    return np.array(((f, 0, 0, 0),
                     (0, f, 0, 0),
                     (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
                     (0, 0, -1, 0)))


def _orthographic(width=20.0, near=1.0, far=100.0):
    return np.array(((2 / width, 0, 0, 0),
                     (0, 2 / width, 0, 0),
                     (0, 0, -2 / (far - near), -(far + near) / (far - near)),
                     (0, 0, 0, 1)))


@pytest.fixture(params=['SPHERE', 'CYLINDER'])
def quad_shader(gl_context, request):
    from OpenGL import GL
    from OpenGL.GL.shaders import compileShader

    shape = request.param
    source = (Path(__file__).parents[1] / 'src' / 'vertexShader.txt').read_text()
    placement = 'USE_SHIFT_AND_SCALE' if shape == 'SPHERE' else 'USE_INSTANCING'
    source = source.replace('#version 330 core',
                            '#version 330 core\n#define %s 1\n#define USE_IMPOSTOR_%s 1\n'
                            % (placement, shape), 1)
    shader = compileShader(source, GL.GL_VERTEX_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, shader)
    name = ctypes.create_string_buffer(b'gl_Position')
    names = (ctypes.POINTER(ctypes.c_char) * 1)(ctypes.cast(name, ctypes.POINTER(ctypes.c_char)))
    GL.glTransformFeedbackVaryings(program, 1, names, GL.GL_INTERLEAVED_ATTRIBS)
    GL.glLinkProgram(program)
    assert GL.glGetProgramiv(program, GL.GL_LINK_STATUS), GL.glGetProgramInfoLog(program)

    # Drawing requires a complete framebuffer even with rasterization off.
    fbo = GL.glGenFramebuffers(1)
    rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, 1, 1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)

    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    result = GL.glGenBuffers(1)
    nbytes = 6 * 4 * 4
    GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, result)
    GL.glBufferData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
    GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, result)
    GL.glUseProgram(program)
    GL.glEnable(GL.GL_RASTERIZER_DISCARD)

    def uniform_matrix(name, m):
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, name), 1, True,
                              np.asarray(m, np.float32))

    def quad(start, end, radius, projection):
        '''Return clip coordinates of the 6 quad vertices.'''
        if shape == 'SPHERE':
            GL.glVertexAttrib4f(4, *start, radius)
        else:
            m = _cylinder_placement(start, end, radius)
            for column in range(4):
                GL.glVertexAttrib4f(5 + column, *m[:, column])
            GL.glUniform3f(GL.glGetUniformLocation(program, 'impostor_cylinder'), 1.0, 0.0, 0.0)
        uniform_matrix('model_view_matrix', np.eye(4))
        uniform_matrix('projection_matrix', projection)
        GL.glBeginTransformFeedback(GL.GL_POINTS)
        GL.glDrawArrays(GL.GL_POINTS, 0, 6)
        GL.glEndTransformFeedback()
        data = GL.glGetBufferSubData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, nbytes)
        return np.frombuffer(data, np.float32).reshape((6, 4)).astype(float)

    yield shape, quad

    GL.glDisable(GL.GL_RASTERIZER_DISCARD)
    GL.glUseProgram(0)
    GL.glBindVertexArray(0)
    GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0)
    GL.glDeleteBuffers(1, [result])
    GL.glDeleteVertexArrays(1, [vao])
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(1, [fbo])
    GL.glDeleteRenderbuffers(1, [rb])
    GL.glDeleteProgram(program)
    GL.glDeleteShader(shader)


def _cylinder_placement(start, end, radius):
    '''Placement of a unit height cylinder from start to end, like half-bond placements.'''
    start, end = np.array(start, float), np.array(end, float)
    z = end - start
    axis = z / np.linalg.norm(z)
    x = np.cross(axis, (0, 1, 0) if abs(axis[1]) < 0.9 else (1, 0, 0))
    x *= radius / np.linalg.norm(x)
    y = np.cross(axis, x)
    m = np.eye(4)
    m[:3, :3] = np.column_stack((x, y, z))
    m[:3, 3] = (start + end) / 2
    return m


def _surface_points(shape, start, end, radius):
    a = np.linspace(0, 2 * np.pi, 181)
    if shape == 'SPHERE':
        u = np.linspace(-1, 1, 91)
        uu, aa = np.meshgrid(u, a)
        s = np.sqrt(1 - uu * uu)
        p = np.column_stack(((s * np.cos(aa)).ravel(), (s * np.sin(aa)).ravel(), uu.ravel()))
        return np.array(start) + radius * p
    m = _cylinder_placement(start, end, radius)
    t = np.linspace(-0.5, 0.5, 91)
    tt, aa = np.meshgrid(t, a)
    p = np.column_stack((np.cos(aa).ravel(), np.sin(aa).ravel(), tt.ravel(), np.ones(tt.size)))
    return (p @ m.T)[:, :3]


def _near_z(projection):
    p = projection
    return (p[2, 3] + p[3, 3]) / (-p[3, 2] - p[2, 2])


def _assert_covers(shape, quad, start, end, radius, projection):
    clip = quad(start, end, radius, projection)
    pts = _surface_points(shape, start, end, radius)
    pts = pts[pts[:, 2] <= _near_z(projection)]	# Surface beyond the near plane is drawn.
    if len(pts) == 0:
        assert (clip[:, 3] <= 0).all() or np.ptp(clip[:, :2] / clip[:, 3:4], axis=0).max() == 0
        return
    assert (clip[:, 3] > 0).all(), 'quad crosses the eye plane'
    ndc = clip[:, :2] / clip[:, 3:4]
    corners = ndc[[0, 1, 2, 5]]	# Quad corners in order around the quad
    h = np.column_stack((pts, np.ones(len(pts)))) @ projection.T
    sp = h[:, :2] / h[:, 3:4]
    for i in range(4):
        e = corners[(i + 1) % 4] - corners[i]
        d = sp - corners[i]
        cross = e[0] * d[:, 1] - e[1] * d[:, 0]
        assert (cross >= -1e-5).all(), 'surface point outside quad edge %d' % i


CASES = [
    # start, end (cylinder only), radius
    ((0, 0, -20), (1, 1, -24), 1.5),	# In front, on axis
    ((6, -4, -15), (8, -1, -18), 1.0),	# Off axis
    ((0, 0, -1.5), (0, 0, -6), 1.0),	# Straddles near plane
    ((0.5, 0, 0.2), (0.5, 0, -5), 1.0),	# Crosses the eye plane
    ((2, 1, 3), (3, 1, 0.5), 1.0),	# Crosses the eye plane, mostly behind
    ((0, 0, 0.3), (0, 0, -8), 0.5),	# Points at the camera
]


@pytest.mark.parametrize('start, end, radius', CASES)
def test_perspective_quad_covers_surface(quad_shader, start, end, radius):
    shape, quad = quad_shader
    _assert_covers(shape, quad, start, end, radius, _perspective())


@pytest.mark.parametrize('start, end, radius', CASES[:3])
def test_orthographic_quad_covers_surface(quad_shader, start, end, radius):
    shape, quad = quad_shader
    _assert_covers(shape, quad, start, end, radius, _orthographic())


def test_quad_behind_eye_is_clipped(quad_shader):
    shape, quad = quad_shader
    clip = quad((0, 0, 5), (0, 0, 8), 1.0, _perspective())
    assert (clip[:, 3] <= 0).all()


def test_crossing_quad_is_bounded(quad_shader):
    # A shape crossing the eye plane gets a finite quad, not the whole viewport.
    shape, quad = quad_shader
    clip = quad((6, 0, 0), (6, 0, -3), 2.0, _perspective())
    assert (clip[:, 3] > 0).all()
    ndc = clip[:, :2] / clip[:, 3:4]
    assert ndc[:, 0].min() > -1	# Shape is right of view center.
