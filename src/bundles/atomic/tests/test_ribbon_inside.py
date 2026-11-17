"""Inside colors follow helix geometry, independently of ribbon orientation."""

import numpy as np
import pytest

from chimerax.atomic.ribbon import (
    RibbonXSection, RibbonsDrawing, TriangleAccumulator, XSectionManager,
    _ribbon_geometry, _helix_inside_sides, _vertex_inside_colors,
)


def helix_path():
    angles = np.linspace(0, 2 * np.pi, 32, dtype=np.float32)
    centers = np.column_stack((3 * np.cos(angles), 3 * np.sin(angles), angles))
    tangents = np.column_stack((-3 * np.sin(angles), 3 * np.cos(angles), np.ones(32)))
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    # Width direction is the helix axis projected perpendicular to the tangent.
    normals = np.array((0, 0, 1)) - tangents[:, 2, None] * tangents
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return tuple(np.ascontiguousarray(a, dtype=np.float32) for a in (centers, tangents, normals))


def test_ideal_helix_selects_inward_side():
    path = helix_path()
    # The thickness binormal in this fixture points away from the helix axis.
    np.testing.assert_array_equal(_helix_inside_sides(path, 4, [(0, 4)]), -np.ones(4))
    flipped = (path[0], path[1], -path[2])
    np.testing.assert_array_equal(_helix_inside_sides(flipped, 4, [(0, 4)]), np.ones(4))


def triangle_areas(vertices, triangles):
    xyz = vertices[triangles]
    return .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)


def assert_sharp_colors(triangles, inside):
    colors = inside[triangles]
    # No triangle may interpolate between the inside and residue colors.
    assert (colors == colors[:, :1]).all()


@pytest.mark.parametrize('faceted', [False, True])
@pytest.mark.parametrize('arrow', [False, True])
@pytest.mark.parametrize('flipped', [False, True])
def test_surface_mask_excludes_caps_and_unmarked_sections(faceted, arrow, flipped):
    path = helix_path()
    if flipped:
        path = (path[0], path[1], -path[2])
    coords = np.array(((1, .2), (-1, .2), (-1, -.2), (1, -.2)), np.float32)
    section = RibbonXSection(coords=coords, faceted=faceted)
    if arrow:
        section = section.arrow(((1, 1), (.3, 1)))
    coil = RibbonXSection(coords=coords * .2, faceted=faceted)
    front = [coil, section, section, section]
    back = [coil, section, section, coil]
    geometry = TriangleAccumulator()
    _ribbon_geometry(path, [(0, 3)], 4, front, back, geometry, inside_sections=(section,),
                     helix_ranges=[(1, 4)])
    vertices, normals, triangles, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert inside.dtype == np.bool_
    assert inside.any() and (~inside).any()
    assert_sharp_colors(triangles, inside)
    _, _, _, start, end = geometry.triangle_ranges[0]
    assert not inside[start:end].any()  # The first residue uses only coil sections.
    cap_tangents = path[1][[0, 8, 28, 31]]
    caps = np.max(np.abs(normals @ cap_tangents.T), axis=1) > 1 - 1e-6
    assert caps.any()
    assert not inside[caps].any()

    # The seam retessellates the swept sides; original cap triangles and
    # cross-section extents must remain unchanged, including for arrows.
    plain = TriangleAccumulator()
    _ribbon_geometry(path, [(0, 3)], 4, front, back, plain)
    original_vertices, original_normals, original_triangles = plain.vertex_normal_triangle_arrays()
    original_caps = np.max(np.abs(original_normals @ cap_tangents.T), axis=1) > 1 - 1e-6
    cap_triangles = triangles[caps[triangles].all(axis=1)]
    original_cap_triangles = original_triangles[original_caps[original_triangles].all(axis=1)]
    np.testing.assert_allclose(vertices[cap_triangles], original_vertices[original_cap_triangles], atol=1e-6)
    np.testing.assert_allclose(vertices.min(axis=0), original_vertices.min(axis=0))
    np.testing.assert_allclose(vertices.max(axis=0), original_vertices.max(axis=0))
    assert not plain.vertex_normal_triangle_arrays(include_inside=True)[3].any()


def bent_path(theta):
    tangents = np.column_stack((np.zeros(len(theta)), np.sin(theta), np.cos(theta)))
    centers = np.zeros_like(tangents)
    centers[1:] = np.cumsum((tangents[1:] + tangents[:-1]) * .5, axis=0)
    normals = np.tile((1, 0, 0), (len(theta), 1))
    return tuple(np.ascontiguousarray(a, dtype=np.float32) for a in (centers, tangents, normals))


def test_local_curvature_reversal_and_flat_patch_do_not_swap_sides():
    s = np.linspace(0, 1, 64)
    theta = s + .08 * np.sin(8 * np.pi * s)
    theta[24:33] = theta[24]  # A locally straight interval within the helix.
    path = bent_path(theta)
    curvature = path[1][2:] - path[1][:-2]
    local_facing = (np.cross(path[1][1:-1], path[2][1:-1]) * curvature).sum(axis=1)
    assert (local_facing < -.001).any() and (local_facing > .001).any()
    assert (np.abs(local_facing) < 1e-7).any()
    np.testing.assert_array_equal(_helix_inside_sides(path, 8, [(0, 8)]), np.ones(8))
    # Inverting the local frame must choose the opposite parameter side, keeping
    # the same physical face colored.
    flipped = (path[0], path[1], -path[2])
    np.testing.assert_array_equal(_helix_inside_sides(flipped, 8, [(0, 8)]), -np.ones(8))


def test_adjacent_helices_choose_sides_independently():
    path = bent_path(.8 * np.sin(np.pi * np.linspace(0, 1, 64)))
    np.testing.assert_array_equal(_helix_inside_sides(path, 8, [(0, 4), (4, 8)]),
                                  [1, 1, 1, 1, -1, -1, -1, -1])
    straight = bent_path(np.zeros(64))
    assert not _helix_inside_sides(straight, 8, [(0, 8)]).any()


@pytest.mark.parametrize('display_ranges', [[(0, 7)], [(0, 2), (5, 7)]])
def test_distorted_helix_keeps_same_face_even_with_display_gaps(display_ranges):
    s = np.linspace(0, 1, 64)
    theta = s + .08 * np.sin(8 * np.pi * s)
    theta[24:33] = theta[24]
    path = bent_path(theta)
    section = XSectionManager()._make_xs(XSectionManager.STYLE_SQUARE, (1, .2))
    geometry = TriangleAccumulator()
    _ribbon_geometry(path, display_ranges, 8, [section] * 8, [section] * 8, geometry,
                     inside_sections=(section,), helix_ranges=[(0, 8)])
    vertices, normals, triangles, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert_sharp_colors(triangles, inside)
    # Find the path point for each vertex.  In this planar fixture the surface
    # offset is .2 and consecutive centers are about 1 apart, so it is unique.
    distances = ((vertices[:, None, 1:] - path[0][None, :, 1:]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    binormals = np.cross(path[1], path[2])[nearest]
    side = (normals * binormals).sum(axis=1)
    broad_faces = np.abs(side) > .9  # Exclude end caps and narrow edge faces.
    assert broad_faces.any()
    np.testing.assert_array_equal(inside[broad_faces], side[broad_faces] > 0)


@pytest.mark.parametrize('faceted', [False, True])
@pytest.mark.parametrize('arrow', [False, True])
@pytest.mark.parametrize('inside_side', [-1, 1])
def test_material_side_is_continuous_along_entire_face(faceted, arrow, inside_side):
    from chimerax.atomic import _ribbons

    # Width is X, thickness is Y, and the path follows Z.  The selected side
    # must stay colored along the WHOLE face, with no transverse color seam.
    centers = np.zeros((16, 3), np.float32)
    centers[:, 2] = np.linspace(0, 1, 16)
    tangents = np.tile(np.array((0, 0, 1), np.float32), (16, 1))
    normals = np.tile(np.array((1, 0, 0), np.float32), (16, 1))
    coords = np.array(((1, .2), (-1, .2), (-1, -.2), (1, -.2)), np.float32)
    section = RibbonXSection(coords=coords, faceted=faceted)
    if arrow:
        section = section.arrow(((1, 1), (.3, 1)))
    ptr = section._xs_pointer
    geometry = TriangleAccumulator()
    _ribbons.ribbon_extrusions(centers, tangents, normals, [(0, 1)], 2,
                               [ptr] * 2, [ptr] * 2, geometry._geom_cpp,
                               inside_sides=np.full((2, 2), inside_side, np.int32))
    vertices, normals, triangles, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert_sharp_colors(triangles, inside)
    xyz = vertices[triangles]
    surface = np.abs(normals[triangles, 2]).max(axis=1) < .5
    assert not inside[triangles[~surface]].any()  # Caps retain residue colors.
    xyz = xyz[surface]
    actual = inside[triangles[surface, 0]]
    centroids = xyz.mean(axis=1)
    np.testing.assert_array_equal(actual, centroids[:, 1] * inside_side > 0)
    assert not ((xyz[:, :, 1].min(axis=1) < -1e-6)
                & (xyz[:, :, 1].max(axis=1) > 1e-6)).any()

    plain = TriangleAccumulator()
    _ribbons.ribbon_extrusions(centers, tangents,
                               np.tile(np.array((1, 0, 0), np.float32), (16, 1)),
                               [(0, 1)], 2, [ptr] * 2, [ptr] * 2, plain._geom_cpp)
    original_vertices, _, original_triangles = plain.vertex_normal_triangle_arrays()
    np.testing.assert_allclose(triangle_areas(vertices, triangles).sum(),
                               triangle_areas(original_vertices, original_triangles).sum(), rtol=2e-6)

    # Duplicate midplane vertices preserve smooth lighting across the edge.
    boundary = np.flatnonzero((np.abs(vertices[:, 1]) < 1e-6)
                              & (np.abs(normals[:, 2]) < .5))
    assert inside[boundary].any() and (~inside[boundary]).any()
    for index in boundary:
        matches = (np.max(np.abs(vertices - vertices[index]), axis=1) < 1e-6)
        matches &= inside != inside[index]
        assert matches.any()
        assert (np.max(np.abs(normals[matches] - normals[index]), axis=1) < 1e-6).any()


def test_inside_color_survives_recolor_and_clear(test_production_session):
    from chimerax.atomic import AtomicStructure, Residues
    structure = AtomicStructure(test_production_session)
    try:
        residue = structure.new_residue('ALA', 'A', 1)
        residue.ribbon_color = (20, 30, 40, 255)
        drawing = RibbonsDrawing('ribbon', 'test')
        drawing.set_geometry(np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32),
                             np.array(((0, 1, 2), (1, 2, 3)), np.int32))
        drawing.set_triangle_ranges(Residues([residue]), np.array(((0, 0, 2, 0, 4),), np.int32))
        drawing._helix_inside_mask = np.array((True, False, True, False))
        inside = np.array(((255, 0, 0, 128), (0, 255, 0, 255)), np.uint8)
        drawing._helix_inside_colors = inside
        for color in ((20, 30, 40, 255), (0, 0, 255, 200)):
            residue.ribbon_color = color
            drawing.update_ribbon_colors()
            np.testing.assert_array_equal(drawing.vertex_colors[[0, 2]], inside)
            np.testing.assert_array_equal(drawing.vertex_colors[[1, 3]], [color, color])
        drawing.clear()
        assert drawing._helix_inside_mask is None
        assert drawing._helix_inside_colors is None
    finally:
        structure.delete()


def make_structure(session, num_res, *, with_atoms=True):
    from chimerax.atomic import AtomicStructure
    structure = AtomicStructure(session)
    for i in range(num_res):
        residue = structure.new_residue('ALA', 'A', i + 1)
        if with_atoms:
            atom = structure.new_atom('CA', 'C')
            residue.add_atom(atom)
            atom.coord = (i, 0, 0)
    return structure


def test_residue_inside_colors_and_structure_copy(test_production_session):
    structure = make_structure(test_production_session, 4)
    copied = None
    try:
        r = list(structure.residues)
        assert structure.residues.helix_inside_colors.tolist() == [None] * 4
        colors = [None, (1, 2, 3, 4), None, (5, 6, 7, 8)]
        structure.residues.helix_inside_colors = colors
        assert r[1].helix_inside_color == colors[1]
        assert structure.residues.helix_inside_colors.tolist() == colors
        copied = structure.copy()
        assert copied.residues.helix_inside_colors.tolist() == colors
        r[1].helix_inside_color = None
        r[3].delete()
        assert structure.residues.helix_inside_colors.tolist() == [None] * 3
        assert copied.residues.helix_inside_colors.tolist() == colors
        # Cross-section state is independent of residue colors.
        snapshot = structure.ribbon_xs_mgr.take_snapshot(None, None)
        assert not any('color' in key for key in snapshot)
        assert not hasattr(structure.ribbon_xs_mgr, 'helix_inside_color')
        with pytest.raises(ValueError):
            structure.residues.helix_inside_colors = [None]
        with pytest.raises(ValueError):
            r[0].helix_inside_color = (1, 2, 3)
    finally:
        if copied is not None:
            copied.delete()
        structure.delete()


def test_inside_color_scene_snapshots(test_production_session):
    from chimerax.core.state import State
    structure = make_structure(test_production_session, 3)
    try:
        colors = [(1, 2, 3, 4), None, (20, 30, 40, 128)]
        empty = structure.take_snapshot(test_production_session, State.SCENE)
        structure.residues.helix_inside_colors = colors
        snapshot = structure.take_snapshot(test_production_session, State.SCENE)
        structure.restore_scene(empty)
        assert structure.residues.helix_inside_colors.tolist() == [None] * 3
        structure.restore_scene(snapshot)
        assert structure.residues.helix_inside_colors.tolist() == colors
        structure.interpolate_scene(empty, snapshot, .25, switchover=False)
        assert structure.residues.helix_inside_colors.tolist() == [None] * 3
        structure.interpolate_scene(empty, snapshot, .75, switchover=True)
        assert structure.residues.helix_inside_colors.tolist() == colors
    finally:
        structure.delete()


@pytest.mark.parametrize("version", [21, 22])
def test_inside_color_native_session_round_trip(test_production_session, version):
    from chimerax.atomic import AtomicStructure, StructureData
    structure = make_structure(test_production_session, 3, with_atoms=False)
    restored = AtomicStructure(test_production_session)
    try:
        colors = [(1, 2, 3, 4), None, (20, 30, 40, 128)]
        structure.residues.helix_inside_colors = colors
        structure._ses_call("save_setup")
        try:
            data = StructureData.save_state(structure, test_production_session, 0).data
        finally:
            structure._ses_call("save_teardown")
        if version == 21:
            # Reconstruct the previous native format for these atomless residues:
            # two constructor ints, two RGBA colors, and nine residue ints.
            # Version 22 adds a third RGBA color and a final enabled flag.
            rows = data['ints'][5].reshape(3, 24)
            data['ints'][5] = np.concatenate((rows[:, :10], rows[:, 14:23]), axis=1).ravel()
            data['version'] = 21
        StructureData.set_state_from_snapshot(restored, test_production_session, data)
        expected = colors if version == 22 else [None] * 3
        assert restored.residues.helix_inside_colors.tolist() == expected
        assert restored.residues.ribbon_colors.tolist() == structure.residues.ribbon_colors.tolist()
    finally:
        handler = getattr(restored, '_ses_end_handler', None)
        if handler is not None:
            test_production_session.triggers.remove_handler(handler)
            restored._ses_restore_teardown()
        restored.delete()
        structure.delete()


def test_vertex_inside_colors_use_owning_residue():
    # Residue 2's empty range shares a start with residue 1 and must not
    # claim its vertices.  Ranges need not be in vertex order.
    ranges = np.array(((1, 4, 6, 3, 6), (0, 0, 4, 0, 3), (2, 6, 6, 3, 3), (3, 6, 8, 6, 9)),
                      np.int32)
    residue_colors = np.array(((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)), np.uint8)
    inside = np.zeros(9, bool)
    inside[[1, 3, 5, 8]] = True
    np.testing.assert_array_equal(_vertex_inside_colors(inside, ranges, residue_colors)[:, 0],
                                  [0, 1, 1, 3])


def test_only_colored_residues_of_a_helix_get_inside_faces():
    s = np.linspace(0, 1, 64)
    path = bent_path(s)
    section = XSectionManager()._make_xs(XSectionManager.STYLE_SQUARE, (1, .2))
    geometry = TriangleAccumulator()
    colored = np.array((False, False, True, True, False, True, False, False))
    _ribbon_geometry(path, [(0, 7)], 8, [section] * 8, [section] * 8, geometry,
                     inside_sections=(section,), helix_ranges=[(0, 8)],
                     inside_residues=colored)
    vertices, normals, triangles, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert_sharp_colors(triangles, inside)
    for res, ts, te, vs, ve in geometry.triangle_ranges:
        assert inside[vs:ve].any() == colored[res]
    # All colored residues share the helix's side, even if separated.
    whole = TriangleAccumulator()
    _ribbon_geometry(path, [(0, 7)], 8, [section] * 8, [section] * 8, whole,
                     inside_sections=(section,), helix_ranges=[(0, 8)])
    whole_inside = whole.vertex_normal_triangle_arrays(include_inside=True)[3]
    for (res, _, _, vs, ve), (_, _, _, wvs, wve) in zip(geometry.triangle_ranges,
                                                        whole.triangle_ranges):
        if colored[res]:
            np.testing.assert_array_equal(inside[vs:ve], whole_inside[wvs:wve])


@pytest.mark.parametrize('style', [XSectionManager.STYLE_ROUND, XSectionManager.STYLE_PIPING])
def test_curved_cross_sections_and_polymer_offsets(style):
    manager = XSectionManager()
    manager.style_helix = style
    section = manager.xs_helix
    path = helix_path()
    geometry = TriangleAccumulator()
    _ribbon_geometry(path, [(0, 3)], 4, [section] * 4, [section] * 4, geometry,
                     inside_sections=(section,), helix_ranges=[(0, 4)])
    va, na, ta, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert inside.any()
    assert_sharp_colors(ta, inside)
    # An uncolored polymer appended to the same accumulator must not inherit
    # the first polymer's inside mask or residue indices.
    geometry.set_range_offset(4)
    _ribbon_geometry(path, [(0, 3)], 4, [section] * 4, [section] * 4, geometry)
    combined = geometry.vertex_normal_triangle_arrays(include_inside=True)
    np.testing.assert_array_equal(combined[3][:len(inside)], inside)
    assert not combined[3][len(inside):].any()
    np.testing.assert_array_equal(geometry.triangle_ranges[:, 0], np.arange(8))


@pytest.mark.parametrize('style', [XSectionManager.STYLE_ROUND, XSectionManager.STYLE_PIPING])
@pytest.mark.parametrize('scale', [(1, .2), (.2, 1)])
def test_cached_seam_preserves_caps_and_can_color_either_side(style, scale):
    from chimerax.atomic import _ribbons

    manager = XSectionManager()
    manager.params[manager.STYLE_ROUND]['sides'] = 5  # Seam crosses an oval edge.
    section = manager._make_xs(style, scale)
    ptr = section._xs_pointer
    centers = np.zeros((16, 3), np.float32)
    centers[:, 2] = np.linspace(0, 1, 16)
    tangents = np.tile(np.array((0, 0, 1), np.float32), (16, 1))
    normals = np.tile(np.array((1, 0, 0), np.float32), (16, 1))
    results = []
    for side in (0, 1, -1):
        geometry = TriangleAccumulator()
        _ribbons.ribbon_extrusions(centers, tangents, normals, [(0, 1)], 2,
                                   [ptr] * 2, [ptr] * 2, geometry._geom_cpp,
                                   inside_sides=np.full((2, 2), side, np.int32))
        va, na, ta, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
        results.append((va, na, ta, inside))
        caps = np.abs(na[:, 2]) > .9
        assert not inside[caps].any()
        cap_triangles = ta[caps[ta].all(axis=1)]
        if side == 0:
            original_caps = va[cap_triangles]
        else:
            np.testing.assert_allclose(va[cap_triangles], original_caps, atol=1e-6)
            assert_sharp_colors(ta, inside)
    # Reusing the cached seam for the opposite side changes only the mask.
    for positive, negative in zip(results[1][:3], results[2][:3]):
        np.testing.assert_array_equal(positive, negative)
    surface = np.abs(results[1][1][:, 2]) < .5
    assert (results[1][3][surface] != results[2][3][surface]).all()


def test_half_section_assignments_are_independent():
    from chimerax.atomic import _ribbons

    centers = np.zeros((16, 3), np.float32)
    centers[:, 2] = np.linspace(0, 1, 16)
    tangents = np.tile(np.array((0, 0, 1), np.float32), (16, 1))
    normals = np.tile(np.array((1, 0, 0), np.float32), (16, 1))
    section = XSectionManager()._make_xs(XSectionManager.STYLE_SQUARE, (1, .2))
    ptr = section._xs_pointer
    geometry = TriangleAccumulator()
    _ribbons.ribbon_extrusions(centers, tangents, normals, [(0, 1)], 2,
                               [ptr] * 2, [ptr] * 2, geometry._geom_cpp,
                               inside_sides=np.array(((1, 0), (0, -1)), np.int32))
    va, na, ta, inside = geometry.vertex_normal_triangle_arrays(include_inside=True)
    assert_sharp_colors(ta, inside)
    surface = np.abs(na[ta, 2]).max(axis=1) < .5
    centroids = va[ta[surface]].mean(axis=1)
    expected = ((centroids[:, 2] < centers[4, 2]) & (centroids[:, 1] > 0))
    expected |= ((centroids[:, 2] > centers[12, 2]) & (centroids[:, 1] < 0))
    np.testing.assert_array_equal(inside[ta[surface, 0]], expected)


@pytest.mark.parametrize('shape', [(1, 2), (2, 1)])
def test_inside_sides_requires_one_pair_per_residue(shape):
    from chimerax.atomic import _ribbons

    centers, tangents, normals = helix_path()
    section = XSectionManager().xs_helix
    ptr = section._xs_pointer
    geometry = TriangleAccumulator()
    with pytest.raises(TypeError):
        _ribbons.ribbon_extrusions(centers, tangents, normals, [(0, 1)], 2,
                                   [ptr] * 2, [ptr] * 2, geometry._geom_cpp,
                                   inside_sides=np.zeros(shape, np.int32))
