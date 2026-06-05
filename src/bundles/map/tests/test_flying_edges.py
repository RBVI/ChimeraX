import pytest


np = pytest.importorskip("numpy")


def _contour_pair(data, level, *, cap_faces=False):
    import chimerax.map._flying_edges as flying_edges
    import chimerax.map._map as marching_cubes

    fe_vertices, fe_triangles, fe_normals = flying_edges.contour_surface(
        data, level, cap_faces=cap_faces, calculate_normals=True
    )
    mc_vertices, mc_triangles, mc_normals = marching_cubes.contour_surface(
        data, level, cap_faces=cap_faces, calculate_normals=True
    )

    return (
        (fe_vertices, fe_triangles, fe_normals),
        (mc_vertices, mc_triangles, mc_normals),
    )


def _assert_surface_shapes_match(flying_edges_surface, marching_cubes_surface):
    fe_vertices, fe_triangles, fe_normals = flying_edges_surface
    mc_vertices, mc_triangles, mc_normals = marching_cubes_surface

    assert fe_vertices.shape == mc_vertices.shape
    assert fe_triangles.shape == mc_triangles.shape
    assert fe_normals.shape == mc_normals.shape
    assert np.isfinite(fe_vertices).all()
    assert np.isfinite(fe_normals).all()


def test_flying_edges_matches_marching_cubes_for_closed_surface():
    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[1:5, 1:5, 1:5] = 2.0

    flying_edges_surface, marching_cubes_surface = _contour_pair(data, 1.0)

    _assert_surface_shapes_match(flying_edges_surface, marching_cubes_surface)


def test_flying_edges_matches_marching_cubes_on_boundary_with_normals():
    z, y, x = np.indices((6, 6, 6), dtype=np.float32)
    data = x + y + z

    flying_edges_surface, marching_cubes_surface = _contour_pair(
        data, 1.5, cap_faces=False
    )

    _assert_surface_shapes_match(flying_edges_surface, marching_cubes_surface)
