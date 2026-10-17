from types import SimpleNamespace

import numpy as np

from chimerax.graphics.wgpu_prototype import (
    _IMPOSTOR_FRAGMENT_GLSL,
    _IMPOSTOR_SHADOW_FRAGMENT_GLSL,
    _IMPOSTOR_SHADOW_VERTEX_GLSL,
    _IMPOSTOR_VERTEX_GLSL,
    _fit_transform,
    _glsl_fragment_source,
    _multishadow_directions,
    _shadow_ray_uniforms,
    _shadow_view_projection,
    chimerax_view_projection,
    displayed_meshes,
    displayed_volumes,
    highlight_from_drawing,
    mesh_from_drawing,
)
from chimerax.graphics.ral import BackendCapabilities
from chimerax.graphics.view import View


class _Places:
    def __init__(self, translations=((0, 0, 0),)):
        self.translations = np.asarray(translations, dtype=np.float32)

    def __len__(self):
        return len(self.translations)

    def __iter__(self):
        return iter(_TranslationPlace(shift) for shift in self.translations)

    def opengl_matrices(self):
        matrices = np.repeat(
            np.eye(4, dtype=np.float32)[np.newaxis, :, :], len(self), axis=0
        )
        matrices[:, 3, 0:3] = self.translations
        return matrices

    def shift_and_scale_array(self):
        return None

    def masked(self, mask):
        return _Places(self.translations[np.asarray(mask, dtype=bool)])

    def __mul__(self, points):
        if isinstance(points, _Places):
            return _Places(
                [
                    left + right
                    for left in self.translations
                    for right in points.translations
                ]
            )
        return np.concatenate([points + shift for shift in self.translations])


class _TranslationPlace:
    def __init__(self, shift):
        self.shift = np.asarray(shift, dtype=np.float32)

    def __mul__(self, points):
        if isinstance(points, _TranslationPlace):
            return _TranslationPlace(self.shift + points.shift)
        return np.asarray(points) + self.shift

    def inverse(self):
        return _TranslationPlace(-self.shift)

    @property
    def matrix(self):
        matrix = np.zeros((3, 4), dtype=np.float32)
        matrix[:, :3] = np.eye(3, dtype=np.float32)
        matrix[:, 3] = self.shift
        return matrix


class _ShiftScalePlaces:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def __len__(self):
        return len(self.values)

    def shift_and_scale_array(self):
        return self.values

    def opengl_matrices(self):
        matrices = np.zeros((len(self), 4, 4), dtype=np.float32)
        matrices[:, 0, 0] = self.values[:, 3]
        matrices[:, 1, 1] = self.values[:, 3]
        matrices[:, 2, 2] = self.values[:, 3]
        matrices[:, 3, :3] = self.values[:, :3]
        matrices[:, 3, 3] = 1
        return matrices


class _PackedCylinderPlaces:
    def __init__(self, matrices):
        self.matrices = np.asarray(matrices, dtype=np.float32)

    def __len__(self):
        return len(self.matrices)

    def shift_and_scale_array(self):
        return None

    def opengl_matrices(self):
        return self.matrices


class _Drawing:
    name = "map surface"
    color = (64, 128, 255, 255)
    display = True
    display_style = "solid"
    vertex_colors = None
    texture = None
    multitexture = None
    on_top = False
    use_lighting = True

    def __init__(self, vertices, normals, triangles, positions=None, colors=None):
        self.vertices = vertices
        self.normals = normals
        self.triangles = triangles
        self.masked_triangles = triangles
        self._positions = _Places() if positions is None else positions
        self.positions = self._positions
        self._displayed_positions = None
        self.highlighted_positions = None
        self.highlighted_triangles_mask = None
        self.drawing_lineage = (self,)
        self._colors = (
            np.asarray((self.color,), dtype=np.uint8) if colors is None else colors
        )

    def get_scene_positions(self, displayed_only=False):
        return self._positions

    def get_positions(self, displayed_only=False):
        return self._positions

    def get_colors(self, displayed_only=False):
        return self._colors

    def _transparency(self):
        colors = self.vertex_colors if self.vertex_colors is not None else self._colors
        return np.any(colors[:, 3] == 255), np.any(colors[:, 3] < 255)

    def all_drawings(self, displayed_only=False):
        return [self]

    @property
    def highlighted(self):
        positions = self.highlighted_positions
        triangles = self.highlighted_triangles_mask
        return bool(
            (positions is not None and np.any(positions))
            or (triangles is not None and np.any(triangles))
        )

    @property
    def highlighted_displayed_triangles_mask(self):
        return self.highlighted_triangles_mask


def test_non_front_buffer_rectangle_becomes_toggleable_overlay():
    view = SimpleNamespace(
        _render=SimpleNamespace(capabilities=BackendCapabilities()),
        _use_opengl=lambda: True,
        background_color=(0, 0, 0, 0),
        _overlay_rectangle=None,
        redraw_needed=False,
    )

    View.draw_xor_rectangle(view, 1, 2, 3, 4, (5, 6, 7, 8))
    assert view._overlay_rectangle == ((1, 2, 3, 4), (5, 6, 7, 8))

    View.draw_xor_rectangle(view, 1, 2, 3, 4, (5, 6, 7, 8))
    assert view._overlay_rectangle is None


def test_mesh_keeps_gpu_ready_numpy_arrays():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32)
    normals = np.asarray(((0, 0, 1),) * 3, np.float32)
    triangles = np.asarray(((0, 1, 2),), np.int32)

    mesh = mesh_from_drawing(_Drawing(vertices, normals, triangles))

    assert mesh.vertices is vertices
    assert mesh.normals is normals
    assert mesh.triangles is triangles
    assert mesh.reused_vertices
    assert mesh.reused_normals
    assert mesh.reused_triangles
    assert mesh.instance_count == 1
    assert mesh.vertex_colors_are_instance
    assert len(mesh.vertex_colors) == mesh.instance_count


def test_highlight_keeps_only_selected_instances_and_triangles():
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), np.float32
    )
    triangles = np.asarray(((0, 1, 2), (1, 3, 2)), np.int32)
    drawing = _Drawing(
        vertices,
        None,
        triangles,
        positions=_Places(((0, 0, 0), (5, 0, 0), (10, 0, 0))),
    )
    drawing.highlighted_positions = np.asarray((False, True, False))
    drawing.highlighted_triangles_mask = np.asarray((True, False))

    highlight = highlight_from_drawing(drawing)

    np.testing.assert_array_equal(highlight.triangles, ((0, 1, 2),))
    assert len(highlight.instance_matrices) == 1
    np.testing.assert_allclose(highlight.instance_matrices[0, 3, :3], (5, 0, 0))


def test_mesh_generates_normals_when_missing():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32)
    triangles = np.asarray(((0, 1, 2),), np.int32)

    mesh = mesh_from_drawing(_Drawing(vertices, None, triangles))

    np.testing.assert_allclose(mesh.normals, ((0, 0, 1),) * 3)
    assert not mesh.reused_normals


def test_mesh_keeps_all_displayed_instances():
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32)
    normals = np.asarray(((0, 0, 1),) * 3, np.float32)
    triangles = np.asarray(((0, 1, 2),), np.int32)
    positions = _Places(((0, 0, 0), (3, 0, 0)))
    colors = np.asarray(((255, 0, 0, 255), (0, 0, 255, 255)), np.uint8)

    mesh = mesh_from_drawing(
        _Drawing(vertices, normals, triangles, positions=positions, colors=colors)
    )

    assert mesh.instance_count == 2
    np.testing.assert_array_equal(mesh.instance_colors, colors)
    np.testing.assert_allclose(mesh.bounds[0], (0, 0, 0))
    np.testing.assert_allclose(mesh.bounds[1], (4, 1, 0))


def test_sphere_impostors_keep_center_radius_instances_and_true_bounds():
    from chimerax.graphics.renderer import SHADER_IMPOSTOR_SPHERE

    vertices = np.asarray(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
    normals = np.asarray(((0, 0, 1),) * 4, np.float32)
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), np.int32)
    positions = _ShiftScalePlaces(((1, 2, 3, 2), (-4, 0, 1, 0.5)))
    drawing = _Drawing(vertices, normals, triangles, positions=positions)
    drawing._shader_options = lambda: SHADER_IMPOSTOR_SPHERE

    mesh = mesh_from_drawing(drawing)

    assert mesh.impostor_kind == "sphere"
    np.testing.assert_allclose(mesh.impostor_data, positions.values)
    np.testing.assert_allclose(mesh.bounds[0], (-4.5, -0.5, 0.5))
    np.testing.assert_allclose(mesh.bounds[1], (3, 4, 5))


def test_cylinder_impostors_unpack_color_metadata_before_transforming():
    from chimerax.graphics.renderer import SHADER_IMPOSTOR_CYLINDER

    vertices = np.asarray(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
    normals = np.asarray(((0, 0, 1),) * 4, np.float32)
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), np.int32)
    packed = np.eye(4, dtype=np.float32)[np.newaxis, :, :]
    packed[0, 0, 0] = packed[0, 1, 1] = 2
    packed[0, 2, 2] = 6
    packed[0, 3, :3] = (10, 20, 30)
    endpoint_color = np.asarray((12, 34, 56, 178), dtype=np.uint8)
    packed[0, :, 3] = endpoint_color / 255.0
    positions = _PackedCylinderPlaces(packed)
    drawing = _Drawing(vertices, normals, triangles, positions=positions)
    drawing._shader_options = lambda: SHADER_IMPOSTOR_CYLINDER
    drawing.impostor_dashes = 7

    mesh = mesh_from_drawing(drawing)

    assert mesh.impostor_kind == "cylinder"
    assert mesh.impostor_dashes == 7
    np.testing.assert_array_equal(mesh.secondary_colors[0], endpoint_color)
    np.testing.assert_allclose(mesh.impostor_data[0, :, 3], (0, 0, 0, 1))
    np.testing.assert_allclose(mesh.impostor_data[0, 3, :3], (10, 20, 30))
    np.testing.assert_allclose(mesh.bounds[0], (8, 18, 25))
    np.testing.assert_allclose(mesh.bounds[1], (12, 22, 35))


def test_impostor_shaders_use_glsl_and_enable_conservative_depth():
    shader_groups = (
        _IMPOSTOR_VERTEX_GLSL,
        _IMPOSTOR_FRAGMENT_GLSL,
        _IMPOSTOR_SHADOW_VERTEX_GLSL,
        _IMPOSTOR_SHADOW_FRAGMENT_GLSL,
    )
    for shaders in shader_groups:
        assert set(shaders) == {"sphere", "cylinder"}
        assert all(source.startswith("#version 450") for source in shaders.values())
    for shaders in (_IMPOSTOR_VERTEX_GLSL, _IMPOSTOR_SHADOW_VERTEX_GLSL):
        assert all("front_depth * center_clip.w" in source for source in shaders.values())

    for shaders in (_IMPOSTOR_FRAGMENT_GLSL, _IMPOSTOR_SHADOW_FRAGMENT_GLSL):
        for source in shaders.values():
            conservative = _glsl_fragment_source(source, True)
            fallback = _glsl_fragment_source(source, False)
            assert "#define CHIMERAX_CONSERVATIVE_DEPTH 1" in conservative
            assert "layout(depth_greater) out float gl_FragDepth;" in conservative
            assert "#define CHIMERAX_CONSERVATIVE_DEPTH 0" in fallback
            assert "gl_FragDepth =" in fallback
    for source in _IMPOSTOR_FRAGMENT_GLSL.values():
        assert "uniform texture2D shadow_map;" in source
        assert "uniform texture2DArray multishadow_map;" in source
        assert "texture2DShadow" not in source
        assert "textureGrad(" in source
        assert "textureLod(" not in source


def test_mesh_preserves_vertex_colors_and_transparency_pass():
    drawing = _Drawing(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
        np.asarray(((0, 0, 1),) * 3, np.float32),
        np.asarray(((0, 1, 2),), np.int32),
    )
    drawing.vertex_colors = np.asarray(
        ((255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 128)), np.uint8
    )

    mesh = mesh_from_drawing(drawing)

    assert mesh.vertex_colors is drawing.vertex_colors
    assert not mesh.vertex_colors_are_instance
    assert mesh.render_pass == "transparent"
    np.testing.assert_array_equal(mesh.instance_colors, ((255, 255, 255, 255),))


def test_displayed_meshes_returns_all_geometry_in_pass_order():
    geometry = (
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
        np.asarray(((0, 0, 1),) * 3, np.float32),
        np.asarray(((0, 1, 2),), np.int32),
    )
    transparent = _Drawing(*geometry, colors=np.asarray(((255, 0, 0, 100),), np.uint8))
    opaque = _Drawing(*geometry)

    class _Root:
        def all_drawings(self, displayed_only=False):
            return [transparent, opaque]

    meshes = displayed_meshes(_Root())

    assert len(meshes) == 2
    assert [mesh.render_pass for mesh in meshes] == ["opaque", "transparent"]


def test_mesh_and_dot_styles_select_webgpu_topologies():
    geometry = (
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
        np.asarray(((0, 0, 1),) * 3, np.float32),
        np.asarray(((0, 1, 2),), np.int32),
    )
    mesh_drawing = _Drawing(*geometry)
    mesh_drawing.display_style = "mesh"
    mesh_drawing.masked_edges = np.asarray(((0, 1), (1, 2), (2, 0)), np.int32)
    dot_drawing = _Drawing(*geometry)
    dot_drawing.display_style = "dot"

    mesh = mesh_from_drawing(mesh_drawing)
    dots = mesh_from_drawing(dot_drawing)

    assert mesh.primitive_topology == "line-list"
    assert mesh.primitive_count == 3
    assert dots.primitive_topology == "point-list"
    assert dots.primitive_count == 3


def test_fit_transform_centers_and_scales_mesh():
    vertices = np.asarray(((-2, -1, -4), (2, 3, 4)), np.float32)
    matrix = _fit_transform((vertices.min(axis=0), vertices.max(axis=0)))
    homogeneous = np.column_stack((vertices, np.ones(2, np.float32)))
    transformed = (matrix @ homogeneous.T).T

    np.testing.assert_allclose(transformed[:, 0:2].mean(axis=0), (0, 0))
    assert np.max(np.abs(transformed[:, 0:2])) <= 0.9
    assert np.all((transformed[:, 2] >= 0) & (transformed[:, 2] <= 1))


class _IdentityPlace:
    def inverse(self, is_orthonormal=False):
        return self

    def opengl_matrix(self):
        return np.eye(4, dtype=np.float32)


class _IdentityCamera:
    def get_position(self, view_num):
        return _IdentityPlace()

    def projection_matrix(self, near_far, view_num, window_size):
        return np.eye(4, dtype=np.float32)


class _IdentityView:
    camera = _IdentityCamera()

    def near_far_distances(self, camera, view_num):
        return 1, 100


def test_chimerax_projection_converts_opengl_depth_to_webgpu_depth():
    matrix = chimerax_view_projection(_IdentityView(), (800, 600))
    points = np.asarray(((0, 0, -1, 1), (0, 0, 1, 1)), np.float32)
    transformed = (matrix @ points.T).T

    np.testing.assert_allclose(transformed[:, 2], (0, 1))


def test_shadow_projection_fits_scene_bounds_in_webgpu_clip_space():
    class _Mesh:
        bounds = (
            np.asarray((-2, -1, -4), np.float32),
            np.asarray((2, 3, 4), np.float32),
        )

    matrix = _shadow_view_projection([_Mesh()], (0.577, -0.577, -0.577))
    lower, upper = _Mesh.bounds
    corners = np.asarray(
        [
            (x, y, z, 1)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        np.float32,
    )
    projected = (matrix @ corners.T).T

    assert np.all(projected[:, :2] >= -1)
    assert np.all(projected[:, :2] <= 1)
    assert np.all(projected[:, 2] >= 0)
    assert np.all(projected[:, 2] <= 1)


def test_shadow_ray_uniforms_pack_inverse_and_viewport_for_impostors():
    matrix = np.asarray(
        ((2, 0, 0, -3), (0, 4, 0, 5), (0, 0, 0.5, 0.25), (0, 0, 0, 1)),
        np.float32,
    )

    uniforms = _shadow_ray_uniforms(matrix, (512, 256))
    packed_matrix = uniforms[:16].reshape((4, 4)).T
    packed_inverse = uniforms[16:32].reshape((4, 4)).T

    assert len(uniforms) == 36
    np.testing.assert_allclose(packed_matrix, matrix)
    np.testing.assert_allclose(packed_inverse @ packed_matrix, np.eye(4), atol=1e-6)
    np.testing.assert_allclose(uniforms[32:], (512, 256, 0, 0))


def test_multishadow_directions_are_unit_length_and_sphere_balanced():
    directions = _multishadow_directions(64)

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(directions.mean(axis=0), 0, atol=0.05)


def test_displayed_volume_retains_contiguous_voxels_and_normalizes_colormap():
    drawing = _Drawing(
        np.asarray(((0, 0, 0), (1, 1, 1)), np.float32),
        None,
        np.asarray(((0, 1, 1),), np.int32),
    )

    class _Texture:
        dimension = 3

        def __init__(self, data):
            self.data = data

    voxels = np.arange(2 * 3 * 4, dtype=np.uint16).reshape((2, 3, 4))
    drawing.texture = _Texture(voxels)
    drawing.colormap = _Texture(
        np.asarray(((0, 0, 0, 0), (255, 255, 255, 255)), np.uint8)
    )
    drawing.colormap_range = (1000, 2000)
    drawing.ray_step = 0.01

    volumes = displayed_volumes(drawing)

    assert len(volumes) == 1
    assert volumes[0].values is voxels
    np.testing.assert_allclose(
        volumes[0].colormap_range, (1000 / 65535, 65535 / 1000)
    )
    np.testing.assert_allclose(volumes[0].world_to_texture, np.eye(4))
    assert not volumes[0].raycast_lighting


def test_displayed_volume_uses_renderer_neutral_materializer_data():
    drawing = _Drawing(
        np.asarray(((0, 0, 0), (1, 1, 1)), np.float32),
        None,
        np.asarray(((0, 1, 1),), np.int32),
    )

    class _Texture:
        dimension = 3
        data = None

    voxels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape((2, 3, 4))
    colormap = np.asarray(((0, 0), (255, 255)), np.uint8)
    drawing.texture = _Texture()
    drawing.colormap = _Texture()
    drawing.volume_data = voxels
    drawing.colormap_data = colormap

    class _VolumeProvider:
        def __init__(self):
            self.materialize_count = 0

        def all_drawings(self, displayed_only=False):
            return [self]

        def materialize_volume_drawing(self):
            self.materialize_count += 1
            return drawing

    provider = _VolumeProvider()
    volume = displayed_volumes(provider)[0]

    assert provider.materialize_count == 1
    assert volume.values is voxels
    assert volume.colormap is colormap


def test_displayed_raycast_volume_keeps_crop_transform_and_render_options():
    drawing = _Drawing(
        np.asarray(((0, 0, 0), (1, 1, 1)), np.float32),
        None,
        np.asarray(((0, 1, 1),), np.int32),
    )

    class _Texture:
        dimension = 3

        def __init__(self, data):
            self.data = data

    drawing.texture = _Texture(np.ones((4, 4, 4), np.float32))
    drawing.colormap = _Texture(
        np.asarray(((0, 0), (255, 255)), np.uint8)
    )
    drawing.vertex_to_texture = _TranslationPlace((0.25, 0, 0))
    drawing.raycast_lighting = True
    drawing.ray_step = 0.02
    drawing.maximum_intensity_projection = True
    drawing.alpha_correction = True
    drawing.linear_interpolation = False

    volume = displayed_volumes(drawing)[0]

    assert volume.raycast_lighting
    assert volume.alpha_correction
    assert volume.maximum_intensity_projection
    assert not volume.linear_interpolation
    assert volume.colormap_component_count == 2
    np.testing.assert_allclose(volume.texture_bounds[0], (0.25, 0, 0))
    np.testing.assert_allclose(volume.texture_bounds[1], (1.25, 1, 1))
    np.testing.assert_allclose(volume.world_to_texture[:3, 3], (0.25, 0, 0))


def test_displayed_volume_keeps_plane_sampling_metadata():
    drawing = _Drawing(
        np.asarray(((0, 0, 0), (1, 1, 0)), np.float32),
        None,
        np.asarray(((0, 1, 1),), np.int32),
    )

    class _Texture:
        dimension = 3

        def __init__(self, data):
            self.data = data

    drawing.texture = _Texture(np.ones((1, 8, 8), np.uint8))
    drawing.vertex_to_texture = _TranslationPlace((0, 0, 0))
    drawing.image_mode = "orthoplanes"
    drawing.orthoplane_positions = (0.25, 0.5, 0.75)
    drawing.orthoplanes_shown = (True, False, True)
    drawing.tilted_slab_plane = (0.0, 1.0, 0.0, -0.25)
    drawing.tilted_slab_range = (-1.0, 2.0)

    volume = displayed_volumes(drawing)[0]

    assert volume.image_mode == "orthoplanes"
    assert volume.orthoplane_positions == (0.25, 0.5, 0.75)
    assert volume.orthoplanes_shown == (True, False, True)
    assert volume.tilted_slab_plane == (0.0, 1.0, 0.0, -0.25)
    assert volume.tilted_slab_range == (-1.0, 2.0)
    assert volume.texture_bounds[1][2] > volume.texture_bounds[0][2]
