from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from chimerax.graphics import opengl


@pytest.fixture
def depth_cull_scene(monkeypatch):
    gl = Mock()
    gl.glGetIntegerv.side_effect = lambda name: {
        gl.GL_SAMPLES: 0, gl.GL_DEPTH_FUNC: gl.GL_LESS}[name]
    gl.glGetDoublev.return_value = (0.0, 1.0)
    gl.glGetBooleanv.return_value = True
    gl.glIsEnabled.side_effect = lambda name: name == gl.GL_DEPTH_TEST
    monkeypatch.setattr(opengl, 'GL', gl)
    texture = Mock()
    texture_factory = Mock(return_value=texture)
    monkeypatch.setattr(opengl, 'Texture', texture_factory)
    snapshot = SimpleNamespace(width=128, height=96, depth_texture=texture, delete=Mock())
    monkeypatch.setattr(opengl, 'Framebuffer', Mock(return_value=snapshot))
    source = SimpleNamespace(width=128, height=96, activate=Mock())
    r = SimpleNamespace(recording_opengl=False, opengl_context=object(),
                        current_framebuffer=lambda: source)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    drawings = [SimpleNamespace(_impostor_to_draw=lambda r: object())]
    return SimpleNamespace(cull=opengl.ImpostorDepthCull(), r=r, gl=gl, texture=texture,
                           source=source, snapshot=snapshot,
                           texture_factory=texture_factory, drawings=drawings)


def test_depth_prepass_is_copied_for_each_view(depth_cull_scene):
    s = depth_cull_scene
    for _ in range(2):
        s.cull.begin(s.r, s.drawings)
        assert s.cull.active
        s.texture.bind_texture.assert_called_with(s.cull.texture_unit)
        s.cull.end()
        assert not s.cull.active
        s.texture.unbind_texture.assert_called_with(s.cull.texture_unit)
    assert s.gl.glCopyTexSubImage2D.call_count == 2
    s.texture_factory.assert_called_once()


def test_resize_releases_old_snapshot(depth_cull_scene):
    s = depth_cull_scene
    s.cull.begin(s.r, s.drawings)
    s.cull.end()
    s.source.width, s.source.height = 256, 192
    s.cull.begin(s.r, s.drawings)
    s.snapshot.delete.assert_called_once()
    s.texture.initialize_depth.assert_called_with((256, 192), depth_compare_mode=False)
    assert s.texture_factory.call_count == 2
    s.cull.end()


@pytest.mark.parametrize('skip', ['recording', 'no_impostors', 'msaa', 'reverse_depth', 'stencil'])
def test_unsupported_passes_do_not_cull(depth_cull_scene, skip):
    s = depth_cull_scene
    if skip == 'recording':
        s.r.recording_opengl = True
    elif skip == 'no_impostors':
        s.drawings = [SimpleNamespace(_impostor_to_draw=lambda r: None)]
    elif skip == 'msaa':
        s.gl.glGetIntegerv.side_effect = lambda name: 4
    elif skip == 'reverse_depth':
        s.gl.glGetDoublev.return_value = (1.0, 0.0)
    elif skip == 'stencil':
        s.gl.glIsEnabled.side_effect = lambda name: True
    s.cull.begin(s.r, s.drawings)
    assert not s.cull.active
    s.texture_factory.assert_not_called()
    s.gl.glCopyTexSubImage2D.assert_not_called()


@pytest.mark.parametrize('platform', ['darwin', 'linux', 'win32'])
def test_depth_culling_is_only_on_macos(depth_cull_scene, monkeypatch, platform):
    s = depth_cull_scene
    monkeypatch.setattr(sys, 'platform', platform)
    s.cull.begin(s.r, s.drawings)
    assert s.cull.active == (platform == 'darwin')
    s.cull.end()


def test_only_impostor_shaders_get_depth_cull():
    r = SimpleNamespace(enable_capabilities=0, disable_capabilities=0,
                        impostor_depth_cull=SimpleNamespace(active=True),
                        _opengl_shader=lambda options: options)
    for name in opengl.shader_options:
        setattr(r, name, getattr(opengl.Render, name))
    for shape in (r.SHADER_IMPOSTOR_SPHERE, r.SHADER_IMPOSTOR_CYLINDER):
        assert opengl.Render.shader(r, shape) & r.SHADER_IMPOSTOR_DEPTH_CULL
        for other_depth in (r.SHADER_DEPTH_TEXTURE, r.SHADER_ALPHA_DEPTH, r.SHADER_VOLUME_RAYCASTING):
            assert not opengl.Render.shader(r, shape | other_depth) & r.SHADER_IMPOSTOR_DEPTH_CULL
    assert not opengl.Render.shader(r, r.SHADER_LIGHTING) & r.SHADER_IMPOSTOR_DEPTH_CULL
    r.impostor_depth_cull.active = False
    assert not opengl.Render.shader(r, r.SHADER_IMPOSTOR_SPHERE) & r.SHADER_IMPOSTOR_DEPTH_CULL
