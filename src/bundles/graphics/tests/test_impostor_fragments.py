'''
Render impostors with the real vertex and fragment shaders and compare the
covered pixels and depths with a double precision ray cast.

Requires a desktop OpenGL context, otherwise the tests are skipped.
'''

from pathlib import Path

import numpy as np
import pytest

SIZE = 64


@pytest.fixture(scope='module')
def renderer():
    from Qt.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
    from Qt.QtWidgets import QApplication
    from OpenGL import GL

    app = QApplication.instance() or QApplication([])
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
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

    # Float depth so that depth readback does not limit the comparison.
    fbo = GL.glGenFramebuffers(1)
    color_rb, depth_rb = GL.glGenRenderbuffers(2)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color_rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, SIZE, SIZE)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT32F, SIZE, SIZE)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, color_rb)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb)
    assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
    GL.glViewport(0, 0, SIZE, SIZE)
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    GL.glEnable(GL.GL_DEPTH_TEST)

    programs = {}
    yield _Renderer(programs)

    for p in programs.values():
        GL.glDeleteProgram(p)
    GL.glBindVertexArray(0)
    GL.glDeleteVertexArrays(1, [vao])
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(1, [fbo])
    GL.glDeleteRenderbuffers(2, [color_rb, depth_rb])
    context.doneCurrent()
    surface.destroy()
    assert app is not None


class _Renderer:
    def __init__(self, programs):
        self._programs = programs

    def _program(self, defines):
        from OpenGL import GL
        from OpenGL.GL.shaders import compileProgram, compileShader
        p = self._programs.get(defines)
        if p is None:
            src = Path(__file__).parents[1] / 'src'
            header = '#version 330 core\n' + ''.join('#define %s 1\n' % d for d in defines)
            vs, fs = [(src / name).read_text().replace('#version 330 core\n', header, 1)
                      for name in ('vertexShader.txt', 'fragmentShader.txt')]
            p = compileProgram(compileShader(vs, GL.GL_VERTEX_SHADER),
                               compileShader(fs, GL.GL_FRAGMENT_SHADER), validate = False)
            self._programs[defines] = p
        return p

    def render(self, shape, start, end, radius, projection, clip_plane = None,
               prepass = False, depth_func = 'LESS'):
        '''Return (covered, window depth) arrays indexed [row from bottom, column].'''
        from OpenGL import GL
        defines = ['USE_IMPOSTOR_' + shape, 'USE_SHIFT_AND_SCALE']
        if clip_plane is not None:
            defines.append('USE_CLIP_PLANES')
        if prepass:
            defines.append('USE_IMPOSTOR_DEPTH_PREPASS')
        p = self._program(tuple(defines))
        GL.glUseProgram(p)

        def matrix(name, m):
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(p, name), 1, True,
                                  np.asarray(m, np.float32))
        matrix('model_view_matrix', np.eye(4))
        matrix('projection_matrix', projection)
        GL.glUniform4f(GL.glGetUniformLocation(p, 'color'), 1, 1, 1, 1)
        if clip_plane is not None:
            matrix('model_matrix', np.eye(4))
            matrix('inverse_view_matrix', np.eye(4))
            GL.glUniform1i(GL.glGetUniformLocation(p, 'num_clip_planes'), 1)
            GL.glUniform4f(GL.glGetUniformLocation(p, 'clip_planes'), *clip_plane)
        GL.glVertexAttrib4f(4, *start, radius)

        GL.glClearColor(0, 0, 0, 0)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDepthFunc(getattr(GL, 'GL_' + depth_func))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glUseProgram(0)

        rgba = np.empty((SIZE, SIZE, 4), np.uint8)
        GL.glReadPixels(0, 0, SIZE, SIZE, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
        depth = np.empty((SIZE, SIZE), np.float32)
        GL.glReadPixels(0, 0, SIZE, SIZE, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, depth)
        return rgba[:, :, 3] > 0, depth.astype(float)


def _perspective(near, far, half_width):
    '''Symmetric perspective projection, half_width is tan(field of view / 2).'''
    f = 1 / half_width
    return np.array(((f, 0, 0, 0),
                     (0, f, 0, 0),
                     (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
                     (0, 0, -1, 0)))


def _pixel_rays(projection):
    c = (np.arange(SIZE) + 0.5) / SIZE * 2 - 1
    x, y = np.meshgrid(c, c)		# [row from bottom, column]
    d = np.stack((x / projection[0, 0], y / projection[1, 1], -np.ones_like(x)), axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _eye_z(depth, projection):
    ndc = 2 * depth - 1
    return -projection[2, 3] / (ndc + projection[2, 2])


def _reference(shape, start, end, radius, projection, clip_plane = None, margin = 0.05):
    '''
    Ray cast in double precision.  Returns (covered, eye z, robust) where robust
    marks pixels at least margin (relative to radius) from every visibility
    decision, so float32 rounding in the shader cannot change the result.
    '''
    d = _pixel_rays(projection)
    p = projection
    near = (p[2, 3] + p[3, 3]) / (p[3, 2] + p[2, 2])
    far = p[2, 3] / (p[2, 2] + 1)
    start, end = np.array(start, float), np.array(end, float)
    c = start
    b = d @ c
    rho2 = c @ c - b * b
    disc = radius * radius - rho2
    root = np.sqrt(np.maximum(disc, 0))
    roots = [b - root, b + root]
    inside = [np.ones_like(b, bool)] * 2
    margins = [disc / radius**2]

    hit = disc >= 0
    covered = np.zeros(d.shape[:2], bool)
    z = np.full(d.shape[:2], np.nan)
    for t, ok in zip(roots, inside):
        pt = t[..., None] * d
        visible = hit & ok & (t > 0) & (pt[..., 2] <= -near) & (pt[..., 2] >= -far)
        margins.append((pt[..., 2] + far) / radius)
        if clip_plane is not None:
            plane = pt @ clip_plane[:3] + clip_plane[3]
            visible &= plane >= 0
            margins.append(plane / radius)
        take = visible & ~covered
        z[take] = pt[..., 2][take]
        covered |= take
    robust = np.all([np.abs(m) >= margin for m in margins], axis=0)
    return covered, z, robust


def _plane(normal, point):
    n = np.array(normal, float)
    n /= np.linalg.norm(n)
    return np.append(n, -n @ np.array(point, float))


def _assert_matches(renderer, shape, start, end, radius, projection, clip_plane = None,
                    depth_func = 'LESS', z_tolerance = 0.01):
    covered, depth = renderer.render(shape, start, end, radius, projection,
                                     clip_plane = clip_plane, depth_func = depth_func)
    ref_covered, ref_z, robust = _reference(shape, start, end, radius, projection, clip_plane)
    assert ref_covered[robust].sum() > 50, 'test scene covers too few pixels'
    assert (covered[robust] == ref_covered[robust]).all(), 'covered pixels differ'
    both = robust & ref_covered
    z = _eye_z(depth[both], projection)
    assert np.abs(z - ref_z[both]).max() <= z_tolerance * radius
    return ref_covered, robust


@pytest.mark.parametrize('shape', ['SPHERE'])
def test_thin_shape_far_from_eye(renderer, shape):
    # A 0.2 radius atom at 1000 from the eye.  The squared distance has a
    # float32 spacing near r^2, so a textbook discriminant loses the surface.
    projection = _perspective(990, 1010, half_width = 3e-4)
    _assert_matches(renderer, shape, (0.05, -0.03, -1000), None, 0.2, projection)


@pytest.mark.parametrize('shape', ['SPHERE'])
def test_clip_plane_reveals_back_surface(renderer, shape):
    # An oblique plane removes the front surface but not the back surface,
    # which must be drawn as for triangles.
    projection = _perspective(1, 100, half_width = 0.25)
    start, end, radius = (0.3, -0.2, -10), None, 1.0
    center = start
    plane = _plane((0.5, 0.2, -1), center)
    ref_covered, robust = _assert_matches(renderer, shape, start, end, radius, projection,
                                          clip_plane = plane)
    _, ref_z, _ = _reference(shape, start, end, radius, projection)
    _, clipped_z, _ = _reference(shape, start, end, radius, projection, plane)
    back = robust & ref_covered & (clipped_z < ref_z - 0.5 * radius)
    assert back.sum() > 20, 'test scene shows too little back surface'


def test_hit_beyond_far_plane_is_not_drawn(renderer):
    # With the front removed, the back of the sphere straddles the far plane.
    # Depth at or behind the far plane passes GL_LEQUAL against a cleared
    # depth buffer, so the shader must discard it.
    projection = _perspective(1, 20, half_width = 0.06)
    center = (0, 0, -19.5)
    plane = _plane((0.1, 0, -1), center)
    ref_covered, robust = _assert_matches(renderer, 'SPHERE', center, None, 1.0, projection,
                                          clip_plane = plane, depth_func = 'LEQUAL')
    no_far = _perspective(1, 1000, half_width = 0.06)
    behind_far, _, _ = _reference('SPHERE', center, None, 1.0, no_far, plane)
    assert (robust & behind_far & ~ref_covered).sum() > 20, 'too few hits beyond the far plane'


@pytest.mark.parametrize('shape', ['SPHERE'])
def test_depth_prepass_slack(renderer, shape):
    # The depth prepass is pushed back by a few depth units so that the
    # lit pass, computed by another shader variant, passes GL_LEQUAL.
    projection = _perspective(1, 100, half_width = 0.25)
    args = (shape, (0.3, -0.2, -10), (1.5, 0.4, -11), 1.0, projection)
    covered, depth = renderer.render(*args)
    pre_covered, pre_depth = renderer.render(*args, prepass = True)
    assert covered.sum() > 100
    assert (pre_covered == covered).all()
    units = (pre_depth[covered] - depth[covered]) * (1 << 24)
    assert units.min() >= 31 and units.max() <= 33
