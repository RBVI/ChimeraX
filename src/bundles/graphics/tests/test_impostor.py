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
    d.set_cylinder_impostor(0.5, caps = True, dashes = 3)
    assert d._impostor_to_draw(_Renderer()) == ('cylinder', 0.5, True, 3)


def test_clear_impostor():
    d = _sphere_drawing()
    d.clear_impostor()
    assert d._impostor_to_draw(_Renderer()) is None
