import pytest

from chimerax.graphics import Drawing, Render


class _Renderer:
    '''Stand-in with the Render attributes used to choose impostor drawing.'''
    SHADER_STEREO_360 = Render.SHADER_STEREO_360

    def __init__(self, use_impostors=True, enable_capabilities=0):
        self.use_impostors = use_impostors
        self.enable_capabilities = enable_capabilities


def _sphere_drawing():
    from chimerax.geometry.sphere import sphere_triangulation
    va, ta = sphere_triangulation(80)
    d = Drawing('spheres')
    d.set_geometry(va, va, ta)
    d.set_sphere_impostor()
    return d


def test_solid_drawing_uses_impostor():
    d = _sphere_drawing()
    assert d._impostor_to_draw(_Renderer()) == ('sphere', 1, False, 0)


def test_no_impostor_by_default():
    d = Drawing('plain')
    assert d._impostor_to_draw(_Renderer()) is None


def test_impostors_turned_off():
    d = _sphere_drawing()
    assert d._impostor_to_draw(_Renderer(use_impostors=False)) is None


@pytest.mark.parametrize('style', [Drawing.Mesh, Drawing.Dot])
def test_mesh_and_dot_styles_draw_triangles(style):
    d = _sphere_drawing()
    d.display_style = style
    assert d._impostor_to_draw(_Renderer()) is None


def test_triangle_mask_draws_triangles():
    from numpy import ones
    d = _sphere_drawing()
    d.triangle_mask = ones((len(d.triangles),), bool)
    assert d._impostor_to_draw(_Renderer()) is None


def test_vertex_colors_draw_triangles():
    from numpy import full, uint8
    d = _sphere_drawing()
    d.vertex_colors = full((len(d.vertices), 4), 255, uint8)
    assert d._impostor_to_draw(_Renderer()) is None


def test_stereo_360_draws_triangles():
    d = _sphere_drawing()
    r = _Renderer(enable_capabilities=Render.SHADER_STEREO_360)
    assert d._impostor_to_draw(r) is None


def test_impostor_keeps_triangles():
    # Export, picking, bounds and sessions use the triangles.
    d = _sphere_drawing()
    assert d.number_of_triangles() == len(d.triangles)
    assert d.bounds() is not None


def test_cylinder_impostor():
    from chimerax.surface import cylinder_geometry
    va, na, ta = cylinder_geometry(nc = 10, caps = False, height = 0.5)
    d = Drawing('cylinders')
    d.set_geometry(va, na, ta)
    d.set_cylinder_impostor(0.5, caps = True, dashes = 3.0)
    imp = d._impostor_to_draw(_Renderer())
    assert imp == ('cylinder', 0.5, True, 3)
    assert type(imp[3]) is int
    with pytest.raises(ValueError):
        d.set_cylinder_impostor(0.5, dashes = -1)


def test_clear_impostor():
    d = _sphere_drawing()
    d.clear_impostor()
    assert d._impostor_to_draw(_Renderer()) is None


@pytest.mark.parametrize('impostor_cap', [Render.SHADER_IMPOSTOR_SPHERE, Render.SHADER_IMPOSTOR_CYLINDER])
def test_multishadow_sets_shadow_depth_for_impostor_without_normals(impostor_cap):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from chimerax.graphics.opengl import Multishadow

    ms = Multishadow(SimpleNamespace())
    ms._multishadow_view_transforms = [None] * 4
    ms._multishadow_depth = 12.5
    ms._max_multishadows = 64
    shader = SimpleNamespace(
        capabilities=Render.SHADER_LIGHTING | Render.SHADER_MULTISHADOW | impostor_cap,
        set_integer=Mock(),
        set_float=Mock(),
    )
    ms._set_multishadow_shader_variables(shader)
    shader.set_float.assert_called_once_with('shadow_depth', 12.5)


def test_toggling_impostors_invalidates_multishadow_cache():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from chimerax.graphics.opengl import Multishadow

    lp = SimpleNamespace(multishadow=4, multishadow_map_size=512, multishadow_depth_bias=0.005)
    mat = SimpleNamespace(transparent_cast_shadows=False, meshes_cast_shadows=False)
    r = SimpleNamespace(lighting=lp, material=mat, use_impostors=True)
    ms = Multishadow(r)
    ms._multishadow_current_params = (4, 512, 0.005, False, False, True)
    ms._multishadow_transforms = [None] * 4
    ms._multishadow_directions = lambda: [None] * 4
    ms._bind_depth_texture = Mock()

    assert ms.use_multishadow_map([]) is True
    ms._bind_depth_texture.assert_called_once()

    r.use_impostors = False
    # Cache invalidated; empty drawings have no shadow bounds so returns False without binding stale map.
    assert ms.use_multishadow_map([]) is False
    assert ms._bind_depth_texture.call_count == 1


def _shader_chooser(**attrs):
    from types import SimpleNamespace
    from chimerax.graphics import opengl
    attrs = dict(enable_capabilities=0, disable_capabilities=0, impostor_depth_prepass=False,
                 _opengl_shader=lambda options: options) | attrs
    r = SimpleNamespace(**attrs)
    for name in opengl.shader_options:
        setattr(r, name, getattr(Render, name))
    return r


def test_only_impostor_prepass_shaders_get_depth_slack():
    r = _shader_chooser(impostor_depth_prepass=True)
    for shape in (Render.SHADER_IMPOSTOR_SPHERE, Render.SHADER_IMPOSTOR_CYLINDER):
        assert Render.shader(r, shape) & Render.SHADER_IMPOSTOR_DEPTH_PREPASS
    assert not Render.shader(r, Render.SHADER_LIGHTING) & Render.SHADER_IMPOSTOR_DEPTH_PREPASS
    r.impostor_depth_prepass = False
    assert not Render.shader(r, Render.SHADER_IMPOSTOR_SPHERE) & Render.SHADER_IMPOSTOR_DEPTH_PREPASS


@pytest.mark.parametrize('prepass', [False, True])
def test_draw_depth_sets_prepass_only_while_drawing(monkeypatch, prepass):
    from chimerax.graphics import drawing
    r = _shader_chooser(SHADER_LIGHTING=Render.SHADER_LIGHTING)
    r.disable_shader_capabilities = lambda c: setattr(r, 'disable_capabilities', c)
    seen = []
    monkeypatch.setattr(drawing, 'draw_opaque', lambda r, d: seen.append(r.impostor_depth_prepass))
    drawing.draw_depth(r, [], prepass=prepass)
    assert seen == [prepass]
    assert r.impostor_depth_prepass is False
    assert r.disable_capabilities == 0
