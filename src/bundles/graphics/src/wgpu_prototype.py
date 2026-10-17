# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# === UCSF ChimeraX Copyright ===

"""Experimental wgpu-py renderer and native ChimeraX canvas support.

The renderer remains isolated from :mod:`opengl`.  It is shared by the main
WGPU graphics widget and the optional standalone benchmark window.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class MeshData:
    """CPU arrays accepted by the prototype renderer."""

    vertices: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    instance_matrices: np.ndarray
    instance_colors: np.ndarray
    bounds: tuple
    source_name: str
    reused_vertices: bool
    reused_normals: bool
    reused_triangles: bool
    vertex_colors: np.ndarray = None
    vertex_colors_are_instance: bool = False
    primitive_topology: str = "triangle-list"
    render_pass: str = "opaque"
    use_lighting: bool = True
    casts_shadows: bool = True
    accept_shadow: bool = True
    accept_multishadow: bool = True
    impostor_kind: str = None
    impostor_data: np.ndarray = None
    secondary_colors: np.ndarray = None
    impostor_dashes: int = 0

    @property
    def triangle_count(self):
        return self.triangles.size // 3 if self.primitive_topology == "triangle-list" else 0

    @property
    def primitive_count(self):
        divisor = {"triangle-list": 3, "line-list": 2, "point-list": 1}[
            self.primitive_topology
        ]
        return self.triangles.size // divisor

    @property
    def instance_count(self):
        return len(self.instance_matrices)

    @property
    def rendered_triangle_count(self):
        return self.triangle_count * self.instance_count


@dataclass
class HighlightData:
    """Selected triangles and instances used to make the highlight mask."""

    vertices: np.ndarray
    triangles: np.ndarray
    instance_matrices: np.ndarray
    source_name: str


@dataclass
class VolumeData:
    """One 3-D NumPy texture and the scene bounds it occupies."""

    values: np.ndarray
    colormap: np.ndarray
    component_count: int
    colormap_component_count: int
    colormap_range: tuple
    world_to_texture: np.ndarray
    texture_bounds: tuple
    color: tuple
    ray_step: float
    raycast_lighting: bool
    maximum_intensity_projection: bool
    alpha_correction: bool
    linear_interpolation: bool
    image_mode: str
    orthoplane_positions: tuple
    orthoplanes_shown: tuple
    tilted_slab_plane: tuple
    tilted_slab_range: tuple
    source_name: str


def _materialize_volume_drawings(root_drawing):
    """Ask volume providers for their current renderer-neutral drawing data."""

    prepared = []
    for drawing in tuple(root_drawing.all_drawings(displayed_only=True)):
        materialize = getattr(drawing, "materialize_volume_drawing", None)
        if materialize is not None:
            result = materialize()
            if result is not None:
                prepared.append(result)
    return prepared


def _contiguous_array(values, dtype, columns):
    a = np.asarray(values)
    reused = a.dtype == np.dtype(dtype) and a.flags.c_contiguous
    if not reused:
        a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim != 2 or a.shape[1] != columns:
        raise ValueError("expected an N by %d array, got %s" % (columns, a.shape))
    return a, reused


def _vertex_normals(vertices, triangles):
    normals = np.zeros(vertices.shape, dtype=np.float32)
    tri = triangles.reshape((-1, 3)).astype(np.intp, copy=False)
    edges1 = vertices[tri[:, 1]] - vertices[tri[:, 0]]
    edges2 = vertices[tri[:, 2]] - vertices[tri[:, 0]]
    face_normals = np.cross(edges1, edges2)
    np.add.at(normals, tri[:, 0], face_normals)
    np.add.at(normals, tri[:, 1], face_normals)
    np.add.at(normals, tri[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 0
    normals[nonzero] /= lengths[nonzero, np.newaxis]
    normals[~nonzero, 2] = 1
    return normals


def _drawing_impostor_kind(drawing):
    """Return the analytic primitive selected by a Drawing's shader flags."""

    shader_options = getattr(drawing, "_shader_options", None)
    if shader_options is None:
        return None
    options = shader_options()
    from .renderer import SHADER_IMPOSTOR_CYLINDER, SHADER_IMPOSTOR_SPHERE
    if options & SHADER_IMPOSTOR_SPHERE:
        return "sphere"
    if options & SHADER_IMPOSTOR_CYLINDER:
        return "cylinder"
    return None


def _parent_opengl_matrices(drawing):
    parent = getattr(drawing, "parent", None)
    if parent is None:
        return np.eye(4, dtype=np.float32)[np.newaxis, :, :]
    positions = parent.get_scene_positions(displayed_only=True)
    return np.ascontiguousarray(positions.opengl_matrices(), dtype=np.float32)


def _sphere_impostor_instances(drawing, positions):
    """Flatten parent placements and local center/radius records to world space."""

    local = positions.shift_and_scale_array()
    if local is None:
        raise ValueError("sphere impostors require shift-and-scale positions")
    local = np.asarray(local, dtype=np.float32)
    parent_matrices = _parent_opengl_matrices(drawing)
    instances = np.empty((len(parent_matrices) * len(local), 4), dtype=np.float32)
    offset = 0
    homogeneous_centers = np.column_stack(
        (local[:, :3], np.ones(len(local), dtype=np.float32))
    )
    for parent_opengl in parent_matrices:
        parent = parent_opengl.T
        count = len(local)
        instances[offset:offset + count, :3] = (
            parent @ homogeneous_centers.T
        ).T[:, :3]
        parent_scale = max(
            float(np.linalg.norm(parent[:3, axis])) for axis in range(3)
        )
        instances[offset:offset + count, 3] = local[:, 3] * parent_scale
        offset += count
    return instances


def _cylinder_impostor_instances(drawing, positions):
    """Return affine world matrices and unpacked second-endpoint colors."""

    packed = np.ascontiguousarray(positions.opengl_matrices(), dtype=np.float32)
    # Bonds encode endpoint two's RGBA in the bottom row.  It is shader data,
    # not the projective row of an affine matrix.
    secondary_colors = np.clip(
        np.rint(packed[:, :, 3] * 255.0), 0, 255
    ).astype(np.uint8)
    local = packed.copy()
    local[:, :3, 3] = 0.0
    local[:, 3, 3] = 1.0

    parent_matrices = _parent_opengl_matrices(drawing)
    count = len(parent_matrices) * len(local)
    instances = np.empty((count, 4, 4), dtype=np.float32)
    colors = np.empty((count, 4), dtype=np.uint8)
    offset = 0
    for parent_opengl in parent_matrices:
        parent = parent_opengl.T
        for matrix, color in zip(local, secondary_colors):
            instances[offset] = (parent @ matrix.T).T
            colors[offset] = color
            offset += 1
    return instances, colors


def _impostor_bounds(kind, data):
    if kind == "sphere":
        centers, radii = data[:, :3], data[:, 3:4]
        return (centers - radii).min(axis=0), (centers + radii).max(axis=0)

    matrices = data.transpose((0, 2, 1))
    centers = matrices[:, :3, 3]
    half_axes = 0.5 * matrices[:, :3, 2]
    radii = np.maximum(
        np.linalg.norm(matrices[:, :3, 0], axis=1),
        np.linalg.norm(matrices[:, :3, 1], axis=1),
    )[:, np.newaxis]
    endpoints = np.concatenate((centers - half_axes, centers + half_axes), axis=0)
    expanded = np.concatenate((radii, radii), axis=0)
    return (endpoints - expanded).min(axis=0), (endpoints + expanded).max(axis=0)


def _shadow_ray_uniforms(matrix, viewport_size):
    """Pack a light transform and its inverse for analytic shadow casters."""

    matrix = np.asarray(matrix, dtype=np.float32)
    width, height = viewport_size
    return np.concatenate(
        (
            np.ascontiguousarray(matrix.T).reshape(-1),
            np.ascontiguousarray(np.linalg.inv(matrix).T, dtype=np.float32).reshape(-1),
            np.asarray((width, height, 0.0, 0.0), dtype=np.float32),
        )
    )


def mesh_from_drawing(drawing, render_pass=None):
    """Return GPU-ready arrays and all displayed instances of *drawing*.

    ChimeraX map surfaces normally already have contiguous float32 vertices and
    normals and contiguous int32 triangles.  Those objects are retained here,
    not converted through Python lists or copied merely to change int32 to
    uint32 (the bit representation of a non-negative index is identical).
    """

    vertices, reused_vertices = _contiguous_array(drawing.vertices, np.float32, 3)

    style = getattr(drawing, "display_style", "solid")
    if style == "mesh":
        indices_in = drawing.masked_edges
        columns = 2
        primitive_topology = "line-list"
    elif style == "dot":
        triangles_in = getattr(drawing, "masked_triangles", None)
        if triangles_in is None:
            triangles_in = drawing.triangles
        indices_in = np.unique(np.asarray(triangles_in).reshape(-1)).reshape((-1, 1))
        columns = 1
        primitive_topology = "point-list"
    else:
        indices_in = getattr(drawing, "masked_triangles", None)
        if indices_in is None:
            indices_in = drawing.triangles
        columns = 3
        primitive_topology = "triangle-list"
    triangles, reused_triangles = _contiguous_array(indices_in, np.int32, columns)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("triangle indices are outside the vertex array")

    normals_in = drawing.normals
    if normals_in is None:
        if primitive_topology == "triangle-list":
            normals = _vertex_normals(vertices, triangles)
        else:
            normals = np.zeros(vertices.shape, dtype=np.float32)
            normals[:, 2] = 1
        reused_normals = False
    else:
        normals, reused_normals = _contiguous_array(normals_in, np.float32, 3)
        if len(normals) != len(vertices):
            raise ValueError("vertex and normal array lengths differ")

    local_positions = drawing.get_positions(displayed_only=True)
    if len(local_positions) == 0:
        raise ValueError("drawing has no displayed positions")
    impostor_kind = _drawing_impostor_kind(drawing)
    impostor_data = secondary_colors = None
    if impostor_kind == "sphere":
        impostor_data = _sphere_impostor_instances(drawing, local_positions)
        instance_matrices = np.repeat(
            np.eye(4, dtype=np.float32)[np.newaxis, :, :],
            len(impostor_data), axis=0,
        )
    elif impostor_kind == "cylinder":
        impostor_data, secondary_colors = _cylinder_impostor_instances(
            drawing, local_positions
        )
        instance_matrices = impostor_data
    else:
        positions = drawing.get_scene_positions(displayed_only=True)
        instance_matrices = np.ascontiguousarray(
            positions.opengl_matrices(), dtype=np.float32
        )
    instance_count = len(instance_matrices)
    vertex_colors_in = getattr(drawing, "vertex_colors", None)
    if vertex_colors_in is None:
        vertex_colors = np.full((instance_count, 4), 255, dtype=np.uint8)
        vertex_colors_are_instance = True
        colors = np.asarray(drawing.get_colors(displayed_only=True), dtype=np.uint8)
    else:
        vertex_colors, _ = _contiguous_array(vertex_colors_in, np.uint8, 4)
        vertex_colors_are_instance = False
        colors = np.full((instance_count, 4), 255, dtype=np.uint8)
    if len(colors) != instance_count:
        # Parent drawings can multiply a child's scene positions without
        # multiplying its color array.  Repeat the child's color sequence in
        # the same order for this prototype.
        colors = np.resize(colors, (instance_count, 4))
    instance_colors = np.ascontiguousarray(colors, dtype=np.uint8)
    bounds = (
        _impostor_bounds(impostor_kind, impostor_data)
        if impostor_kind else _instance_bounds(vertices, positions)
    )
    if render_pass is None:
        if getattr(drawing, "on_top", False):
            render_pass = "last"
        else:
            if hasattr(drawing, "_transparency"):
                _any_opaque, any_transparent = drawing._transparency()
            else:
                any_transparent = bool(
                    np.any(vertex_colors[:, 3] < 255)
                    or np.any(instance_colors[:, 3] < 255)
                )
            render_pass = "transparent" if any_transparent else "opaque"
    return MeshData(
        vertices,
        normals,
        triangles,
        instance_matrices,
        instance_colors,
        bounds,
        getattr(drawing, "name", "unnamed drawing"),
        reused_vertices,
        reused_normals,
        reused_triangles,
        vertex_colors,
        vertex_colors_are_instance,
        primitive_topology,
        render_pass,
        bool(getattr(drawing, "use_lighting", True)),
        bool(getattr(drawing, "casts_shadows", True)),
        bool(getattr(drawing, "accept_shadow", True)),
        bool(getattr(drawing, "accept_multishadow", True)),
        impostor_kind,
        impostor_data,
        secondary_colors,
        int(getattr(drawing, "impostor_dashes", 0)),
    )


def displayed_meshes(root_drawing):
    """Return every displayed non-textured geometry drawing in pass order."""

    pass_order = {"opaque": 0, "transparent": 1, "last": 2}
    meshes = []
    for drawing_order, drawing in enumerate(
        root_drawing.all_drawings(displayed_only=True)
    ):
        vertices = getattr(drawing, "vertices", None)
        triangles = getattr(drawing, "triangles", None)
        if vertices is None or triangles is None or len(triangles) == 0:
            continue
        if len(drawing.get_scene_positions(displayed_only=True)) == 0:
            continue
        # Textures and multitextures need their own bind-group layouts.  Keep
        # them out of the untextured pass instead of rendering misleading
        # white geometry while that pipeline is added.
        if getattr(drawing, "texture", None) is not None or getattr(
            drawing, "multitexture", None
        ):
            continue
        mesh = mesh_from_drawing(drawing)
        meshes.append((pass_order[mesh.render_pass], drawing_order, mesh))
    meshes.sort(key=lambda item: item[:2])
    return [item[2] for item in meshes]


def highlight_from_drawing(drawing):
    """Return the displayed, selected portion of *drawing*, if any."""

    if not getattr(drawing, "highlighted", False):
        return None
    vertices, _reused = _contiguous_array(drawing.vertices, np.float32, 3)
    triangles = np.asarray(drawing.triangles)
    triangle_mask = drawing.highlighted_displayed_triangles_mask
    if triangle_mask is not None:
        triangles = triangles[np.asarray(triangle_mask, dtype=bool)]
    triangles, _reused = _contiguous_array(triangles, np.int32, 3)
    if triangles.size == 0:
        return None

    positions = drawing.positions
    position_mask = getattr(drawing, "_displayed_positions", None)
    highlighted_positions = drawing.highlighted_positions
    if highlighted_positions is not None:
        highlighted_positions = np.asarray(highlighted_positions, dtype=bool)
        position_mask = (
            highlighted_positions
            if position_mask is None
            else np.logical_and(position_mask, highlighted_positions)
        )
    if position_mask is not None:
        positions = positions.masked(position_mask)
    if len(positions) == 0:
        return None

    # A drawing's selected-position mask applies to its own instances.  Parent
    # placements still multiply every selected instance into scene space.
    lineage = getattr(drawing, "drawing_lineage", (drawing,))
    for parent in reversed(lineage[:-1]):
        positions = parent.get_positions(displayed_only=True) * positions
    matrices = np.ascontiguousarray(positions.opengl_matrices(), dtype=np.float32)
    return HighlightData(
        vertices,
        triangles,
        matrices,
        getattr(drawing, "name", "unnamed drawing"),
    )


def displayed_highlights(root_drawing):
    """Return selected geometry from all displayed drawings."""

    highlights = []
    for drawing in root_drawing.all_drawings(displayed_only=True):
        vertices = getattr(drawing, "vertices", None)
        triangles = getattr(drawing, "triangles", None)
        if vertices is None or triangles is None or len(triangles) == 0:
            continue
        highlight = highlight_from_drawing(drawing)
        if highlight is not None:
            highlights.append(highlight)
    return highlights


def displayed_volumes(root_drawing):
    """Return 3-D texture drawings without copying their voxel arrays."""

    prepared = _materialize_volume_drawings(root_drawing)
    volumes = []
    drawings = list(root_drawing.all_drawings(displayed_only=True)) + prepared
    seen = set()
    for drawing in drawings:
        if id(drawing) in seen:
            continue
        seen.add(id(drawing))
        texture = getattr(drawing, "texture", None)
        if texture is None or getattr(texture, "dimension", None) != 3:
            continue
        values = getattr(drawing, "volume_data", None)
        if values is None:
            values = getattr(texture, "data", None)
        if values is None or getattr(drawing, "vertices", None) is None:
            continue
        values = np.asarray(values)
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        positions = drawing.get_scene_positions(displayed_only=True)
        if len(positions) == 0:
            continue
        cmap_texture = getattr(drawing, "colormap", None)
        if hasattr(drawing, "colormap_data"):
            colormap = drawing.colormap_data
        elif cmap_texture is not None:
            colormap = cmap_texture.data
        else:
            colormap = None
        if colormap is not None:
            colormap = np.asarray(colormap)
            if not colormap.flags.c_contiguous:
                colormap = np.ascontiguousarray(colormap)
        component_count = values.shape[3] if values.ndim > 3 else 1
        colormap_component_count = (
            colormap.shape[1] if colormap is not None and colormap.ndim > 1 else 1
        )
        scale = {
            np.dtype(np.int8): 1 / 127,
            np.dtype(np.uint8): 1 / 255,
            np.dtype(np.int16): 1 / 32767,
            np.dtype(np.uint16): 1 / 65535,
            np.dtype(np.float32): 1,
        }.get(values.dtype)
        if scale is None:
            continue
        v0, v1 = getattr(drawing, "colormap_range", (0, 1))
        normalized_range = (
            scale * float(v0),
            1.0 / max(scale * float(v1 - v0), 1e-20),
        )
        raycast_lighting = bool(getattr(drawing, "raycast_lighting", False))
        if raycast_lighting:
            ray_step = float(getattr(drawing, "ray_step", 0.01))
        else:
            ray_step = 1.0 / max(values.shape[:3])
        color = tuple(float(c) / 255 for c in drawing.color)
        vertex_to_texture = getattr(drawing, "vertex_to_texture", None)
        if vertex_to_texture is not None:
            texture_vertices = vertex_to_texture * drawing.vertices
            texture_lower = texture_vertices.min(axis=0)
            texture_upper = texture_vertices.max(axis=0)
            # A one-voxel region has coincident box faces.  Give it one texel
            # of thickness so the fullscreen ray pass has a sample interval.
            texture_size = np.asarray(values.shape[:3][::-1], dtype=np.float32)
            half_texel = 0.5 / np.maximum(texture_size, 1)
            flat_axes = np.isclose(texture_lower, texture_upper)
            texture_lower = texture_lower - flat_axes * half_texel
            texture_upper = texture_upper + flat_axes * half_texel
            texture_bounds = (texture_lower, texture_upper)
        else:
            texture_bounds = (
                np.zeros(3, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )
        for position in positions:
            if vertex_to_texture is not None:
                transform = vertex_to_texture * position.inverse()
                world_to_texture = np.eye(4, dtype=np.float32)
                world_to_texture[:3, :] = transform.matrix
            else:
                local_lower = drawing.vertices.min(axis=0)
                local_upper = drawing.vertices.max(axis=0)
                corners = np.asarray(
                    [
                        (x, y, z)
                        for x in (local_lower[0], local_upper[0])
                        for y in (local_lower[1], local_upper[1])
                        for z in (local_lower[2], local_upper[2])
                    ],
                    dtype=np.float32,
                )
                world_corners = position * corners
                lower, upper = world_corners.min(axis=0), world_corners.max(axis=0)
                scale = 1.0 / np.maximum(upper - lower, 1e-20)
                world_to_texture = np.eye(4, dtype=np.float32)
                world_to_texture[0, 0] = scale[0]
                world_to_texture[1, 1] = scale[1]
                world_to_texture[2, 2] = scale[2]
                world_to_texture[:3, 3] = -scale * lower
            volumes.append(
                VolumeData(
                    values,
                    colormap,
                    component_count,
                    colormap_component_count,
                    normalized_range,
                    world_to_texture,
                    texture_bounds,
                    color,
                    ray_step,
                    raycast_lighting,
                    bool(getattr(drawing, "maximum_intensity_projection", False)),
                    bool(getattr(drawing, "alpha_correction", False)),
                    bool(getattr(drawing, "linear_interpolation", True)),
                    getattr(drawing, "image_mode", "full region"),
                    tuple(getattr(drawing, "orthoplane_positions", (0.5,) * 3)),
                    tuple(getattr(drawing, "orthoplanes_shown", (False,) * 3)),
                    tuple(getattr(drawing, "tilted_slab_plane", (0, 0, 1, 0))),
                    tuple(getattr(drawing, "tilted_slab_range", (0, 1))),
                    getattr(drawing, "name", "3-D texture"),
                )
            )
    return volumes


def largest_displayed_mesh(root_drawing):
    """Choose the displayed drawing with the most rendered triangles."""

    candidates = []
    for drawing in root_drawing.all_drawings(displayed_only=True):
        vertices = getattr(drawing, "vertices", None)
        triangles = getattr(drawing, "triangles", None)
        if vertices is None or triangles is None or len(triangles) == 0:
            continue
        if getattr(drawing, "display_style", "solid") != "solid":
            continue
        instance_count = len(drawing.get_scene_positions(displayed_only=True))
        candidates.append((len(triangles) * instance_count, drawing))
    if not candidates:
        raise ValueError("no displayed solid triangle drawing was found")
    return mesh_from_drawing(max(candidates, key=lambda item: item[0])[1])


_MESH_SHADER = r"""
struct Scene {
    view_projection: mat4x4f,
    light_view_projection: mat4x4f,
    key_light_direction: vec4f,
    key_light_color: vec4f,
    key_light_specular: vec4f,
    fill_light_direction: vec4f,
    fill_light_color: vec4f,
    ambient_color: vec4f,
    camera_position: vec4f,
    shadow_params: vec4f,
};

struct MultiShadow {
    transforms: array<mat4x4f, 64>,
    directions: array<vec4f, 64>,
    params: vec4f,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> multishadow: MultiShadow;
@group(0) @binding(4) var multishadow_map: texture_depth_2d_array;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec4f,
    @location(2) drawing_options: vec3f,
    @location(3) shadow_position: vec4f,
    @location(4) world_position: vec3f,
};

@vertex
fn vs_main(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) model_column_0: vec4f,
    @location(3) model_column_1: vec4f,
    @location(4) model_column_2: vec4f,
    @location(5) model_column_3: vec4f,
    @location(6) color: vec4f,
    @location(7) vertex_color: vec4f,
    @location(8) drawing_options: vec3f,
) -> VertexOutput {
    let model = mat4x4f(
        model_column_0,
        model_column_1,
        model_column_2,
        model_column_3,
    );
    var output: VertexOutput;
    let world_position = model * vec4f(position, 1.0);
    output.position = scene.view_projection * world_position;
    output.shadow_position = scene.light_view_projection * world_position;
    output.world_position = world_position.xyz;
    output.normal = (model * vec4f(normal, 0.0)).xyz;
    output.color = color * vertex_color;
    output.drawing_options = drawing_options;
    return output;
}

@fragment
fn fs_main(
    input: VertexOutput,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4f {
    let n = normalize(input.normal) * select(-1.0, 1.0, front_facing);
    let key_factor = max(-dot(n, normalize(scene.key_light_direction.xyz)), 0.0);
    let fill_factor = max(-dot(n, normalize(scene.fill_light_direction.xyz)), 0.0);
    let shadow_ndc = input.shadow_position.xyz / input.shadow_position.w;
    let shadow_uv = vec2f(
        0.5 * shadow_ndc.x + 0.5,
        0.5 - 0.5 * shadow_ndc.y,
    );
    var shadow_factor = 1.0;
    if scene.shadow_params.x > 0.5
        && input.drawing_options.y > 0.5
        && all(shadow_uv >= vec2f(0.0))
        && all(shadow_uv <= vec2f(1.0))
        && shadow_ndc.z >= 0.0
        && shadow_ndc.z <= 1.0 {
        shadow_factor = textureSampleCompareLevel(
            shadow_map,
            shadow_sampler,
            shadow_uv,
            shadow_ndc.z - scene.shadow_params.y,
        );
    }

    var ambient_shadow = 1.0;
    let shadow_count = u32(multishadow.params.x);
    if shadow_count > 0u && input.drawing_options.z > 0.5 {
        var weighted_visibility = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            if i >= shadow_count {
                break;
            }
            let clip = multishadow.transforms[i] * vec4f(input.world_position, 1.0);
            let ndc = clip.xyz / clip.w;
            let uv = vec2f(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
            var visibility = 1.0;
            if all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))
                && ndc.z >= 0.0 && ndc.z <= 1.0 {
                visibility = textureSampleCompareLevel(
                    multishadow_map,
                    shadow_sampler,
                    uv,
                    i32(i),
                    ndc.z - multishadow.params.y,
                );
            }
            let diffuse = max(-dot(n, multishadow.directions[i].xyz), 0.0);
            weighted_visibility += diffuse * visibility;
        }
        // The mean diffuse response of a uniformly sampled sphere is 0.25.
        // This is the same normalization used by ChimeraX's OpenGL shader.
        ambient_shadow = weighted_visibility / (0.25 * f32(shadow_count));
    }

    let view_direction = normalize(input.world_position - scene.camera_position.xyz);
    let reflected = normalize(reflect(normalize(scene.key_light_direction.xyz), n));
    let specular_factor = pow(
        max(-dot(reflected, view_direction), 0.0),
        scene.key_light_specular.w,
    );
    let diffuse_lighting = ambient_shadow * scene.ambient_color.rgb
        + shadow_factor * key_factor * scene.key_light_color.rgb
        + fill_factor * scene.fill_light_color.rgb;
    let lit_color = input.color.rgb * diffuse_lighting
        + shadow_factor * specular_factor * scene.key_light_specular.rgb;
    let result = mix(input.color.rgb, lit_color, input.drawing_options.x);
    return vec4f(result, input.color.a);
}
"""


_CONSERVATIVE_DEPTH_FEATURE = "shader-early-depth-test"
_CONSERVATIVE_DEPTH_DEFINE = "#define CHIMERAX_CONSERVATIVE_DEPTH 0"


def _glsl_fragment_source(source, conservative_depth):
    """Enable GLSL's depth_greater promise when the device feature is active."""

    value = "1" if conservative_depth else "0"
    return source.replace(
        _CONSERVATIVE_DEPTH_DEFINE,
        "#define CHIMERAX_CONSERVATIVE_DEPTH %s" % value,
        1,
    )


_IMPOSTOR_SPHERE_VERTEX_GLSL = r"""#version 450
layout(std140, set = 0, binding = 0) uniform Scene {
    mat4 view_projection;
    mat4 light_view_projection;
    vec4 key_light_direction;
    vec4 key_light_color;
    vec4 key_light_specular;
    vec4 fill_light_direction;
    vec4 fill_light_color;
    vec4 ambient_color;
    vec4 camera_position;
    vec4 shadow_params;
    mat4 inverse_view_projection;
    vec4 viewport;
};

layout(location = 0) in vec3 position;
layout(location = 6) in vec4 color;
layout(location = 8) in vec4 options;
layout(location = 9) in vec4 sphere;

layout(location = 0) flat out vec4 out_sphere;
layout(location = 1) flat out vec4 out_color;
layout(location = 2) flat out vec4 out_options;

vec2 projected_radius(float radius, float clip_w) {
    vec3 x_row = vec3(
        view_projection[0][0], view_projection[1][0], view_projection[2][0]
    );
    vec3 y_row = vec3(
        view_projection[0][1], view_projection[1][1], view_projection[2][1]
    );
    return 1.15 * radius * vec2(length(x_row), length(y_row))
        / max(abs(clip_w), 1e-6);
}

vec3 view_forward() {
    vec4 near_h = inverse_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 far_h = inverse_view_projection * vec4(0.0, 0.0, 1.0, 1.0);
    return normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
}

void main() {
    vec4 center_clip = view_projection * vec4(sphere.xyz, 1.0);
    vec2 center_ndc = center_clip.xy / center_clip.w;
    vec2 proxy_ndc = center_ndc
        + position.xy * projected_radius(sphere.w, center_clip.w);
    vec3 front = sphere.xyz - sphere.w * view_forward();
    vec4 front_clip = view_projection * vec4(front, 1.0);
    float front_depth = clamp(front_clip.z / front_clip.w, 0.0, 1.0);
    // This front plane is a lower depth bound for every sphere fragment.
    gl_Position = vec4(
        proxy_ndc * center_clip.w, front_depth * center_clip.w, center_clip.w
    );
    out_sphere = sphere;
    out_color = color;
    out_options = options;
}
"""


_IMPOSTOR_CYLINDER_VERTEX_GLSL = r"""#version 450
layout(std140, set = 0, binding = 0) uniform Scene {
    mat4 view_projection;
    mat4 light_view_projection;
    vec4 key_light_direction;
    vec4 key_light_color;
    vec4 key_light_specular;
    vec4 fill_light_direction;
    vec4 fill_light_color;
    vec4 ambient_color;
    vec4 camera_position;
    vec4 shadow_params;
    mat4 inverse_view_projection;
    vec4 viewport;
};

layout(location = 0) in vec3 position;
layout(location = 2) in vec4 model_column_0;
layout(location = 3) in vec4 model_column_1;
layout(location = 4) in vec4 model_column_2;
layout(location = 5) in vec4 model_column_3;
layout(location = 6) in vec4 color;
layout(location = 8) in vec4 options;
layout(location = 9) in vec4 secondary_color;

layout(location = 0) flat out vec4 out_center_radius;
layout(location = 1) flat out vec4 out_axis_half_length;
layout(location = 2) flat out vec4 out_color;
layout(location = 3) flat out vec4 out_secondary_color;
layout(location = 4) flat out vec4 out_options;

vec2 projected_radius(float radius, float clip_w) {
    vec3 x_row = vec3(
        view_projection[0][0], view_projection[1][0], view_projection[2][0]
    );
    vec3 y_row = vec3(
        view_projection[0][1], view_projection[1][1], view_projection[2][1]
    );
    return 1.15 * radius * vec2(length(x_row), length(y_row))
        / max(abs(clip_w), 1e-6);
}

vec3 view_forward() {
    vec4 near_h = inverse_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 far_h = inverse_view_projection * vec4(0.0, 0.0, 1.0, 1.0);
    return normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
}

void main() {
    mat4 model = mat4(
        model_column_0, model_column_1, model_column_2, model_column_3
    );
    vec3 endpoint_0 = (model * vec4(0.0, 0.0, -0.5, 1.0)).xyz;
    vec3 endpoint_1 = (model * vec4(0.0, 0.0, 0.5, 1.0)).xyz;
    vec3 center = 0.5 * (endpoint_0 + endpoint_1);
    vec3 axis_vector = endpoint_1 - endpoint_0;
    float axis_length = length(axis_vector);
    vec3 axis = axis_vector / max(axis_length, 1e-8);
    float radius = max(length(model_column_0.xyz), length(model_column_1.xyz));

    vec4 clip_0 = view_projection * vec4(endpoint_0, 1.0);
    vec4 clip_1 = view_projection * vec4(endpoint_1, 1.0);
    vec4 center_clip = view_projection * vec4(center, 1.0);
    vec2 screen_axis = clip_1.xy / clip_1.w - clip_0.xy / clip_0.w;
    vec2 along = vec2(0.0, 1.0);
    if (length(screen_axis) > 1e-7) {
        along = normalize(screen_axis);
    }
    vec2 perpendicular = vec2(-along.y, along.x);
    vec2 radius_components = max(
        projected_radius(radius, clip_0.w),
        projected_radius(radius, clip_1.w)
    );
    float radius_ndc = max(radius_components.x, radius_components.y);
    vec2 proxy_ndc = center_clip.xy / center_clip.w
        + 0.5 * screen_axis * position.y
        + along * radius_ndc * position.y
        + perpendicular * radius_ndc * position.x;

    vec3 forward = view_forward();
    vec3 front = dot(endpoint_0, forward) < dot(endpoint_1, forward)
        ? endpoint_0 : endpoint_1;
    vec3 radial_forward = forward - dot(forward, axis) * axis;
    if (length(radial_forward) > 1e-8) {
        front -= radius * normalize(radial_forward);
    }
    vec4 front_clip = view_projection * vec4(front, 1.0);
    float front_depth = clamp(front_clip.z / front_clip.w, 0.0, 1.0);
    gl_Position = vec4(
        proxy_ndc * center_clip.w, front_depth * center_clip.w, center_clip.w
    );
    out_center_radius = vec4(center, radius);
    out_axis_half_length = vec4(axis, 0.5 * axis_length);
    out_color = color;
    out_secondary_color = secondary_color;
    out_options = options;
}
"""


_IMPOSTOR_FRAGMENT_GLSL_HEADER = r"""#version 450
#define CHIMERAX_CONSERVATIVE_DEPTH 0
#if CHIMERAX_CONSERVATIVE_DEPTH
layout(depth_greater) out float gl_FragDepth;
#endif

layout(std140, set = 0, binding = 0) uniform Scene {
    mat4 view_projection;
    mat4 light_view_projection;
    vec4 key_light_direction;
    vec4 key_light_color;
    vec4 key_light_specular;
    vec4 fill_light_direction;
    vec4 fill_light_color;
    vec4 ambient_color;
    vec4 camera_position;
    vec4 shadow_params;
    mat4 inverse_view_projection;
    vec4 viewport;
};
// Naga's GLSL frontend represents depth comparison on the sampler and sample
// operation.  Its separate texture declarations use the non-Shadow spelling.
layout(set = 0, binding = 1) uniform texture2D shadow_map;
layout(set = 0, binding = 2) uniform samplerShadow shadow_sampler;
layout(std140, set = 0, binding = 3) uniform MultiShadow {
    mat4 multishadow_transforms[64];
    vec4 multishadow_directions[64];
    vec4 multishadow_params;
};
layout(set = 0, binding = 4) uniform texture2DArray multishadow_map;

layout(location = 0) out vec4 fragment_color;

struct Ray {
    vec3 origin;
    vec3 direction;
};

Ray viewing_ray() {
    vec2 ndc = vec2(
        2.0 * gl_FragCoord.x / viewport.x - 1.0,
        1.0 - 2.0 * gl_FragCoord.y / viewport.y
    );
    vec4 near_h = inverse_view_projection * vec4(ndc, 0.0, 1.0);
    vec4 far_h = inverse_view_projection * vec4(ndc, 1.0, 1.0);
    vec3 near_point = near_h.xyz / near_h.w;
    vec3 far_point = far_h.xyz / far_h.w;
    return Ray(near_point, normalize(far_point - near_point));
}

float fragment_depth(vec3 world_position) {
    vec4 clip = view_projection * vec4(world_position, 1.0);
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

vec4 lit_color(vec3 world_position, vec3 normal, vec4 color, vec4 options) {
    if (options.x < 0.5) {
        return color;
    }
    vec3 n = normalize(normal);
    float key_factor = max(-dot(n, normalize(key_light_direction.xyz)), 0.0);
    float fill_factor = max(-dot(n, normalize(fill_light_direction.xyz)), 0.0);

    vec4 shadow_clip = light_view_projection * vec4(world_position, 1.0);
    vec3 shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    vec2 shadow_uv = vec2(0.5 * shadow_ndc.x + 0.5, 0.5 - 0.5 * shadow_ndc.y);
    float shadow_factor = 1.0;
    if (shadow_params.x > 0.5
        && options.y > 0.5
        && all(greaterThanEqual(shadow_uv, vec2(0.0)))
        && all(lessThanEqual(shadow_uv, vec2(1.0)))
        && shadow_ndc.z >= 0.0 && shadow_ndc.z <= 1.0) {
        shadow_factor = textureGrad(
            sampler2DShadow(shadow_map, shadow_sampler),
            vec3(shadow_uv, shadow_ndc.z - shadow_params.y),
            vec2(0.0),
            vec2(0.0)
        );
    }

    float ambient_shadow = 1.0;
    uint shadow_count = uint(multishadow_params.x);
    if (shadow_count > 0u && options.z > 0.5) {
        float weighted_visibility = 0.0;
        for (uint i = 0u; i < 64u; ++i) {
            if (i >= shadow_count) {
                break;
            }
            vec4 clip = multishadow_transforms[i] * vec4(world_position, 1.0);
            vec3 ndc = clip.xyz / clip.w;
            vec2 uv = vec2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
            float visibility = 1.0;
            if (all(greaterThanEqual(uv, vec2(0.0)))
                && all(lessThanEqual(uv, vec2(1.0)))
                && ndc.z >= 0.0 && ndc.z <= 1.0) {
                visibility = textureGrad(
                    sampler2DArrayShadow(multishadow_map, shadow_sampler),
                    vec4(uv, float(i), ndc.z - multishadow_params.y),
                    vec2(0.0),
                    vec2(0.0)
                );
            }
            float diffuse = max(-dot(n, multishadow_directions[i].xyz), 0.0);
            weighted_visibility += diffuse * visibility;
        }
        ambient_shadow = weighted_visibility / (0.25 * float(shadow_count));
    }

    vec3 view_direction = normalize(world_position - camera_position.xyz);
    vec3 reflected = normalize(reflect(normalize(key_light_direction.xyz), n));
    float specular_factor = pow(
        max(-dot(reflected, view_direction), 0.0), key_light_specular.w
    );
    vec3 diffuse_lighting = ambient_shadow * ambient_color.rgb
        + shadow_factor * key_factor * key_light_color.rgb
        + fill_factor * fill_light_color.rgb;
    vec3 rgb = color.rgb * diffuse_lighting
        + shadow_factor * specular_factor * key_light_specular.rgb;
    return vec4(rgb, color.a);
}

float cap_distance(
    Ray ray, vec3 center, vec3 axis, float half_length, float radius, float cap_sign
) {
    float denominator = dot(ray.direction, axis);
    if (abs(denominator) < 1e-8) {
        return 1e30;
    }
    vec3 cap_center = center + cap_sign * half_length * axis;
    float hit_distance = dot(cap_center - ray.origin, axis) / denominator;
    vec3 radial = ray.origin + hit_distance * ray.direction - cap_center;
    if (hit_distance >= 0.0 && dot(radial, radial) <= radius * radius) {
        return hit_distance;
    }
    return 1e30;
}
"""


_IMPOSTOR_SPHERE_FRAGMENT_GLSL = _IMPOSTOR_FRAGMENT_GLSL_HEADER + r"""
layout(location = 0) flat in vec4 sphere;
layout(location = 1) flat in vec4 color;
layout(location = 2) flat in vec4 options;

void main() {
    Ray ray = viewing_ray();
    vec3 delta = ray.origin - sphere.xyz;
    float b = dot(delta, ray.direction);
    float c = dot(delta, delta) - sphere.w * sphere.w;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        discard;
    }
    float root = sqrt(discriminant);
    float hit_distance = -b - root;
    if (hit_distance < 0.0) {
        hit_distance = -b + root;
    }
    if (hit_distance < 0.0) {
        discard;
    }
    vec3 hit = ray.origin + hit_distance * ray.direction;
    vec3 normal = (hit - sphere.xyz) / sphere.w;
    fragment_color = lit_color(hit, normal, color, options);
    gl_FragDepth = fragment_depth(hit);
}
"""


_IMPOSTOR_CYLINDER_FRAGMENT_GLSL = _IMPOSTOR_FRAGMENT_GLSL_HEADER + r"""
layout(location = 0) flat in vec4 center_radius;
layout(location = 1) flat in vec4 axis_half_length;
layout(location = 2) flat in vec4 color;
layout(location = 3) flat in vec4 secondary_color;
layout(location = 4) flat in vec4 options;

void main() {
    Ray ray = viewing_ray();
    vec3 center = center_radius.xyz;
    float radius = center_radius.w;
    vec3 axis = axis_half_length.xyz;
    float half_length = axis_half_length.w;
    vec3 delta = ray.origin - center;
    float direction_axis = dot(ray.direction, axis);
    float delta_axis = dot(delta, axis);
    vec3 direction_perp = ray.direction - direction_axis * axis;
    vec3 delta_perp = delta - delta_axis * axis;
    float a = dot(direction_perp, direction_perp);
    float b = dot(delta_perp, direction_perp);
    float c = dot(delta_perp, delta_perp) - radius * radius;

    float hit_distance = 1e30;
    vec3 normal = vec3(0.0);
    if (a > 1e-10) {
        float discriminant = b * b - a * c;
        if (discriminant >= 0.0) {
            float side_distance = (-b - sqrt(discriminant)) / a;
            float side_along = delta_axis + side_distance * direction_axis;
            if (side_distance >= 0.0 && abs(side_along) <= half_length) {
                hit_distance = side_distance;
                vec3 side_hit = ray.origin + hit_distance * ray.direction;
                normal = normalize(side_hit - center - side_along * axis);
            }
        }
    }

    float positive_cap = cap_distance(ray, center, axis, half_length, radius, 1.0);
    if (positive_cap < hit_distance) {
        hit_distance = positive_cap;
        normal = axis;
    }
    float negative_cap = cap_distance(ray, center, axis, half_length, radius, -1.0);
    if (negative_cap < hit_distance) {
        hit_distance = negative_cap;
        normal = -axis;
    }
    if (hit_distance >= 1e29) {
        discard;
    }

    vec3 hit = ray.origin + hit_distance * ray.direction;
    float along = dot(hit - center, axis);
    if (options.w > 0.5) {
        float fraction = along / max(2.0 * half_length, 1e-8) + 0.5;
        if (fract(fraction * options.w + 0.75) > 0.5) {
            discard;
        }
    }
    vec4 surface_color = along > 0.0 ? secondary_color : color;
    fragment_color = lit_color(hit, normal, surface_color, options);
    gl_FragDepth = fragment_depth(hit);
}
"""


_IMPOSTOR_VERTEX_GLSL = {
    "sphere": _IMPOSTOR_SPHERE_VERTEX_GLSL,
    "cylinder": _IMPOSTOR_CYLINDER_VERTEX_GLSL,
}
_IMPOSTOR_FRAGMENT_GLSL = {
    "sphere": _IMPOSTOR_SPHERE_FRAGMENT_GLSL,
    "cylinder": _IMPOSTOR_CYLINDER_FRAGMENT_GLSL,
}


_SHADOW_SHADER = r"""
struct ShadowScene {
    light_view_projection: mat4x4f,
};

@group(0) @binding(0) var<uniform> shadow_scene: ShadowScene;

@vertex
fn vs_main(
    @location(0) position: vec3f,
    @location(2) model_column_0: vec4f,
    @location(3) model_column_1: vec4f,
    @location(4) model_column_2: vec4f,
    @location(5) model_column_3: vec4f,
) -> @builtin(position) vec4f {
    let model = mat4x4f(
        model_column_0,
        model_column_1,
        model_column_2,
        model_column_3,
    );
    return shadow_scene.light_view_projection * model * vec4f(position, 1.0);
}
"""


_IMPOSTOR_SHADOW_SPHERE_VERTEX_GLSL = r"""#version 450
layout(std140, set = 0, binding = 0) uniform ShadowScene {
    mat4 light_view_projection;
    mat4 inverse_light_view_projection;
    vec4 viewport;
};

layout(location = 0) in vec3 position;
layout(location = 9) in vec4 sphere;
layout(location = 0) flat out vec4 out_sphere;

vec2 projected_radius(float radius, float clip_w) {
    vec3 x_row = vec3(
        light_view_projection[0][0],
        light_view_projection[1][0],
        light_view_projection[2][0]
    );
    vec3 y_row = vec3(
        light_view_projection[0][1],
        light_view_projection[1][1],
        light_view_projection[2][1]
    );
    return 1.02 * radius * vec2(length(x_row), length(y_row))
        / max(abs(clip_w), 1e-6);
}

vec3 light_forward() {
    vec4 near_h = inverse_light_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 far_h = inverse_light_view_projection * vec4(0.0, 0.0, 1.0, 1.0);
    return normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
}

void main() {
    vec4 center_clip = light_view_projection * vec4(sphere.xyz, 1.0);
    vec2 proxy_ndc = center_clip.xy / center_clip.w
        + position.xy * projected_radius(sphere.w, center_clip.w);
    vec3 front = sphere.xyz - sphere.w * light_forward();
    vec4 front_clip = light_view_projection * vec4(front, 1.0);
    float front_depth = clamp(front_clip.z / front_clip.w, 0.0, 1.0);
    gl_Position = vec4(
        proxy_ndc * center_clip.w, front_depth * center_clip.w, center_clip.w
    );
    out_sphere = sphere;
}
"""


_IMPOSTOR_SHADOW_CYLINDER_VERTEX_GLSL = r"""#version 450
layout(std140, set = 0, binding = 0) uniform ShadowScene {
    mat4 light_view_projection;
    mat4 inverse_light_view_projection;
    vec4 viewport;
};

layout(location = 0) in vec3 position;
layout(location = 2) in vec4 model_column_0;
layout(location = 3) in vec4 model_column_1;
layout(location = 4) in vec4 model_column_2;
layout(location = 5) in vec4 model_column_3;
layout(location = 8) in vec4 options;
layout(location = 0) flat out vec4 out_center_radius;
layout(location = 1) flat out vec4 out_axis_half_length;
layout(location = 2) flat out float out_dashes;

vec2 projected_radius(float radius, float clip_w) {
    vec3 x_row = vec3(
        light_view_projection[0][0],
        light_view_projection[1][0],
        light_view_projection[2][0]
    );
    vec3 y_row = vec3(
        light_view_projection[0][1],
        light_view_projection[1][1],
        light_view_projection[2][1]
    );
    return 1.02 * radius * vec2(length(x_row), length(y_row))
        / max(abs(clip_w), 1e-6);
}

vec3 light_forward() {
    vec4 near_h = inverse_light_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 far_h = inverse_light_view_projection * vec4(0.0, 0.0, 1.0, 1.0);
    return normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
}

void main() {
    mat4 model = mat4(
        model_column_0, model_column_1, model_column_2, model_column_3
    );
    vec3 endpoint_0 = (model * vec4(0.0, 0.0, -0.5, 1.0)).xyz;
    vec3 endpoint_1 = (model * vec4(0.0, 0.0, 0.5, 1.0)).xyz;
    vec3 center = 0.5 * (endpoint_0 + endpoint_1);
    vec3 axis_vector = endpoint_1 - endpoint_0;
    float axis_length = length(axis_vector);
    vec3 axis = axis_vector / max(axis_length, 1e-8);
    float radius = max(length(model_column_0.xyz), length(model_column_1.xyz));

    vec4 clip_0 = light_view_projection * vec4(endpoint_0, 1.0);
    vec4 clip_1 = light_view_projection * vec4(endpoint_1, 1.0);
    vec4 center_clip = light_view_projection * vec4(center, 1.0);
    vec2 screen_axis = clip_1.xy / clip_1.w - clip_0.xy / clip_0.w;
    vec2 along = vec2(0.0, 1.0);
    if (length(screen_axis) > 1e-7) {
        along = normalize(screen_axis);
    }
    vec2 perpendicular = vec2(-along.y, along.x);
    vec2 radius_components = max(
        projected_radius(radius, clip_0.w),
        projected_radius(radius, clip_1.w)
    );
    float radius_ndc = max(radius_components.x, radius_components.y);
    vec2 proxy_ndc = center_clip.xy / center_clip.w
        + 0.5 * screen_axis * position.y
        + along * radius_ndc * position.y
        + perpendicular * radius_ndc * position.x;

    vec3 forward = light_forward();
    vec3 front = dot(endpoint_0, forward) < dot(endpoint_1, forward)
        ? endpoint_0 : endpoint_1;
    vec3 radial_forward = forward - dot(forward, axis) * axis;
    if (length(radial_forward) > 1e-8) {
        front -= radius * normalize(radial_forward);
    }
    vec4 front_clip = light_view_projection * vec4(front, 1.0);
    float front_depth = clamp(front_clip.z / front_clip.w, 0.0, 1.0);
    gl_Position = vec4(
        proxy_ndc * center_clip.w, front_depth * center_clip.w, center_clip.w
    );
    out_center_radius = vec4(center, radius);
    out_axis_half_length = vec4(axis, 0.5 * axis_length);
    out_dashes = options.w;
}
"""


_IMPOSTOR_SHADOW_FRAGMENT_GLSL_HEADER = r"""#version 450
#define CHIMERAX_CONSERVATIVE_DEPTH 0
#if CHIMERAX_CONSERVATIVE_DEPTH
layout(depth_greater) out float gl_FragDepth;
#endif

layout(std140, set = 0, binding = 0) uniform ShadowScene {
    mat4 light_view_projection;
    mat4 inverse_light_view_projection;
    vec4 viewport;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

Ray light_ray() {
    vec2 ndc = vec2(
        2.0 * gl_FragCoord.x / viewport.x - 1.0,
        1.0 - 2.0 * gl_FragCoord.y / viewport.y
    );
    vec4 near_h = inverse_light_view_projection * vec4(ndc, 0.0, 1.0);
    vec4 far_h = inverse_light_view_projection * vec4(ndc, 1.0, 1.0);
    vec3 near_point = near_h.xyz / near_h.w;
    vec3 far_point = far_h.xyz / far_h.w;
    return Ray(near_point, normalize(far_point - near_point));
}

float light_depth(vec3 world_position) {
    vec4 clip = light_view_projection * vec4(world_position, 1.0);
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

float cap_distance(
    Ray ray, vec3 center, vec3 axis, float half_length, float radius, float cap_sign
) {
    float denominator = dot(ray.direction, axis);
    if (abs(denominator) < 1e-8) {
        return 1e30;
    }
    vec3 cap_center = center + cap_sign * half_length * axis;
    float hit_distance = dot(cap_center - ray.origin, axis) / denominator;
    vec3 radial = ray.origin + hit_distance * ray.direction - cap_center;
    if (hit_distance >= 0.0 && dot(radial, radial) <= radius * radius) {
        return hit_distance;
    }
    return 1e30;
}
"""


_IMPOSTOR_SHADOW_SPHERE_FRAGMENT_GLSL = (
    _IMPOSTOR_SHADOW_FRAGMENT_GLSL_HEADER + r"""
layout(location = 0) flat in vec4 sphere;

void main() {
    Ray ray = light_ray();
    vec3 delta = ray.origin - sphere.xyz;
    float b = dot(delta, ray.direction);
    float c = dot(delta, delta) - sphere.w * sphere.w;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        discard;
    }
    float root = sqrt(discriminant);
    float hit_distance = -b - root;
    if (hit_distance < 0.0) {
        hit_distance = -b + root;
    }
    if (hit_distance < 0.0) {
        discard;
    }
    gl_FragDepth = light_depth(ray.origin + hit_distance * ray.direction);
}
"""
)


_IMPOSTOR_SHADOW_CYLINDER_FRAGMENT_GLSL = (
    _IMPOSTOR_SHADOW_FRAGMENT_GLSL_HEADER + r"""
layout(location = 0) flat in vec4 center_radius;
layout(location = 1) flat in vec4 axis_half_length;
layout(location = 2) flat in float dashes;

void main() {
    Ray ray = light_ray();
    vec3 center = center_radius.xyz;
    float radius = center_radius.w;
    vec3 axis = axis_half_length.xyz;
    float half_length = axis_half_length.w;
    vec3 delta = ray.origin - center;
    float direction_axis = dot(ray.direction, axis);
    float delta_axis = dot(delta, axis);
    vec3 direction_perp = ray.direction - direction_axis * axis;
    vec3 delta_perp = delta - delta_axis * axis;
    float a = dot(direction_perp, direction_perp);
    float b = dot(delta_perp, direction_perp);
    float c = dot(delta_perp, delta_perp) - radius * radius;

    float hit_distance = 1e30;
    if (a > 1e-10) {
        float discriminant = b * b - a * c;
        if (discriminant >= 0.0) {
            float side_distance = (-b - sqrt(discriminant)) / a;
            float side_along = delta_axis + side_distance * direction_axis;
            if (side_distance >= 0.0 && abs(side_along) <= half_length) {
                hit_distance = side_distance;
            }
        }
    }
    hit_distance = min(
        hit_distance, cap_distance(ray, center, axis, half_length, radius, 1.0)
    );
    hit_distance = min(
        hit_distance, cap_distance(ray, center, axis, half_length, radius, -1.0)
    );
    if (hit_distance >= 1e29) {
        discard;
    }

    vec3 hit = ray.origin + hit_distance * ray.direction;
    if (dashes > 0.5) {
        float along = dot(hit - center, axis);
        float fraction = along / max(2.0 * half_length, 1e-8) + 0.5;
        if (fract(fraction * dashes + 0.75) > 0.5) {
            discard;
        }
    }
    gl_FragDepth = light_depth(hit);
}
"""
)


_IMPOSTOR_SHADOW_VERTEX_GLSL = {
    "sphere": _IMPOSTOR_SHADOW_SPHERE_VERTEX_GLSL,
    "cylinder": _IMPOSTOR_SHADOW_CYLINDER_VERTEX_GLSL,
}
_IMPOSTOR_SHADOW_FRAGMENT_GLSL = {
    "sphere": _IMPOSTOR_SHADOW_SPHERE_FRAGMENT_GLSL,
    "cylinder": _IMPOSTOR_SHADOW_CYLINDER_FRAGMENT_GLSL,
}


_SILHOUETTE_SHADER = r"""
struct Silhouette {
    color: vec4f,
    size: vec2f,
    thickness: f32,
    depth_jump: f32,
};

@group(0) @binding(0) var<uniform> silhouette: Silhouette;
@group(0) @binding(1) var scene_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let dimensions = vec2i(silhouette.size);
    let pixel = clamp(vec2i(position.xy), vec2i(0), dimensions - vec2i(1));
    let radius = max(1, i32(round(silhouette.thickness)));
    let left = vec2i(max(0, pixel.x - radius), pixel.y);
    let right = vec2i(min(dimensions.x - 1, pixel.x + radius), pixel.y);
    let down = vec2i(pixel.x, max(0, pixel.y - radius));
    let up = vec2i(pixel.x, min(dimensions.y - 1, pixel.y + radius));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    let min_depth = min(
        min(textureLoad(scene_depth, left, 0), textureLoad(scene_depth, right, 0)),
        min(textureLoad(scene_depth, down, 0), textureLoad(scene_depth, up, 0)),
    );
    let max_depth = max(
        max(textureLoad(scene_depth, left, 0), textureLoad(scene_depth, right, 0)),
        max(textureLoad(scene_depth, down, 0), textureLoad(scene_depth, up, 0)),
    );
    if max(max_depth - min_depth, abs(center_depth - min_depth)) < silhouette.depth_jump {
        discard;
    }
    return silhouette.color;
}
"""


_HIGHLIGHT_DEPTH_SHADER = r"""
struct HighlightScene {
    view_projection: mat4x4f,
};

@group(0) @binding(0) var<uniform> scene: HighlightScene;

@vertex
fn vs_main(
    @location(0) position: vec3f,
    @location(2) model_column_0: vec4f,
    @location(3) model_column_1: vec4f,
    @location(4) model_column_2: vec4f,
    @location(5) model_column_3: vec4f,
) -> @builtin(position) vec4f {
    let model = mat4x4f(
        model_column_0,
        model_column_1,
        model_column_2,
        model_column_3,
    );
    return scene.view_projection * model * vec4f(position, 1.0);
}
"""


_HIGHLIGHT_OVERLAY_SHADER = r"""
struct Overlay {
    highlight_color: vec4f,
    rectangle_color: vec4f,
    size: vec2f,
    highlight_thickness: f32,
    rectangle_enabled: f32,
    rectangle: vec4f,
};

@group(0) @binding(0) var<uniform> overlay: Overlay;
@group(0) @binding(1) var selected_depth: texture_depth_2d;
@group(0) @binding(2) var scene_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

fn selected_is_visible(pixel: vec2i) -> bool {
    let selected = textureLoad(selected_depth, pixel, 0);
    if selected >= 0.999999 {
        return false;
    }
    let scene = textureLoad(scene_depth, pixel, 0);
    return selected <= scene + 0.00001;
}

@fragment
fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
    if overlay.rectangle_enabled > 0.5 {
        let thickness = max(1.0, overlay.highlight_thickness);
        let inside_x = position.x >= overlay.rectangle.x - thickness
            && position.x <= overlay.rectangle.z + thickness;
        let inside_y = position.y >= overlay.rectangle.y - thickness
            && position.y <= overlay.rectangle.w + thickness;
        let horizontal = inside_x && (
            abs(position.y - overlay.rectangle.y) <= thickness
            || abs(position.y - overlay.rectangle.w) <= thickness
        );
        let vertical = inside_y && (
            abs(position.x - overlay.rectangle.x) <= thickness
            || abs(position.x - overlay.rectangle.z) <= thickness
        );
        if horizontal || vertical {
            return overlay.rectangle_color;
        }
    }

    let dimensions = vec2i(overlay.size);
    let pixel = clamp(vec2i(position.xy), vec2i(0), dimensions - vec2i(1));
    let radius = max(1, i32(round(overlay.highlight_thickness)));
    var any_selected = false;
    var any_unselected = false;
    for (var y = -radius; y <= radius; y = y + 1) {
        for (var x = -radius; x <= radius; x = x + 1) {
            let sample_pixel = clamp(pixel + vec2i(x, y), vec2i(0), dimensions - vec2i(1));
            if selected_is_visible(sample_pixel) {
                any_selected = true;
            } else {
                any_unselected = true;
            }
        }
    }
    if !(any_selected && any_unselected) {
        discard;
    }
    return overlay.highlight_color;
}
"""


_VOLUME_SHADER = r"""
struct VolumeScene {
    inverse_view_projection: mat4x4f,
    world_to_texture: mat4x4f,
    key_light_direction: vec4f,
    key_light_color: vec4f,
    key_light_specular: vec4f,
    fill_light_direction: vec4f,
    fill_light_color: vec4f,
    ambient_color: vec4f,
    texture_min: vec4f,
    texture_max: vec4f,
    screen_step_colormap: vec4f,
    colormap_range: vec4f,
    modulation_color: vec4f,
    render_options: vec4f,
    plane_positions_mode: vec4f,
    plane_mask: vec4f,
    tilted_slab_plane: vec4f,
    tilted_slab_range: vec4f,
};

@group(0) @binding(0) var<uniform> scene: VolumeScene;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;
@group(0) @binding(3) var color_map: texture_1d<f32>;
@group(0) @binding(4) var color_map_sampler: sampler;
@group(0) @binding(5) var scene_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

fn world_at(ndc: vec2f, depth: f32) -> vec3f {
    let p = scene.inverse_view_projection * vec4f(ndc, depth, 1.0);
    return p.xyz / p.w;
}

fn random_offset(pixel: vec2f) -> f32 {
    return fract(sin(dot(pixel, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn expanded_color(value: vec4f, component_count: f32) -> vec4f {
    if component_count < 1.5 {
        return vec4f(value.rrr, 1.0);
    }
    if component_count < 2.5 {
        return vec4f(value.rrr, value.g);
    }
    if component_count < 3.5 {
        return vec4f(value.rgb, 1.0);
    }
    return value;
}

fn sample_color_at(
    sample_position: vec3f,
    ray_direction: vec3f,
    gradient_offset: f32,
    step_length: f32,
) -> vec4f {
    let voxel = textureSampleLevel(
        volume_texture, volume_sampler, sample_position, 0.0
    );
    var sample_color: vec4f;
    if scene.screen_step_colormap.w > 0.5 {
        let intensity = (voxel.r - scene.colormap_range.x)
            * scene.colormap_range.y;
        sample_color = expanded_color(
            textureSampleLevel(
                color_map, color_map_sampler, intensity, 0.0
            ),
            scene.colormap_range.z,
        ) * scene.modulation_color;
    } else {
        sample_color = expanded_color(
            voxel, scene.render_options.w
        ) * scene.modulation_color;
    }
    if sample_color.a > 0.001 && scene.render_options.x > 0.5 {
        let gradient_texture = vec3f(
            textureSampleLevel(volume_texture, volume_sampler,
                sample_position + vec3f(gradient_offset, 0.0, 0.0), 0.0).r
                - textureSampleLevel(volume_texture, volume_sampler,
                sample_position - vec3f(gradient_offset, 0.0, 0.0), 0.0).r,
            textureSampleLevel(volume_texture, volume_sampler,
                sample_position + vec3f(0.0, gradient_offset, 0.0), 0.0).r
                - textureSampleLevel(volume_texture, volume_sampler,
                sample_position - vec3f(0.0, gradient_offset, 0.0), 0.0).r,
            textureSampleLevel(volume_texture, volume_sampler,
                sample_position + vec3f(0.0, 0.0, gradient_offset), 0.0).r
                - textureSampleLevel(volume_texture, volume_sampler,
                sample_position - vec3f(0.0, 0.0, gradient_offset), 0.0).r,
        );
        var light_color = scene.ambient_color.rgb;
        if length(gradient_texture) > 0.0001 {
            let gradient = normalize((
                transpose(scene.world_to_texture)
                * vec4f(gradient_texture, 0.0)
            ).xyz);
            let key_direction = normalize(scene.key_light_direction.xyz);
            let key_factor = max(-dot(gradient, key_direction), 0.0);
            let fill_factor = max(
                -dot(gradient, normalize(scene.fill_light_direction.xyz)), 0.0
            );
            let reflected = reflect(key_direction, gradient);
            let specular = pow(
                max(-dot(reflected, ray_direction), 0.0),
                scene.key_light_specular.w,
            );
            light_color += key_factor * scene.key_light_color.rgb
                + fill_factor * scene.fill_light_color.rgb;
            sample_color = vec4f(
                sample_color.rgb * light_color
                    + specular * scene.key_light_specular.rgb,
                sample_color.a,
            );
        } else {
            sample_color = vec4f(
                sample_color.rgb * light_color, sample_color.a
            );
        }
    }
    if scene.render_options.z > 0.5 {
        sample_color.a = 1.0 - pow(
            1.0 - sample_color.a, step_length / 0.001
        );
    }
    return sample_color;
}

@fragment
fn fs_main(@builtin(position) fragment_position: vec4f) -> @location(0) vec4f {
    let screen_size = scene.screen_step_colormap.xy;
    let ndc = vec2f(
        2.0 * fragment_position.x / screen_size.x - 1.0,
        1.0 - 2.0 * fragment_position.y / screen_size.y,
    );
    let ray_origin = world_at(ndc, 0.0);
    let ray_far = world_at(ndc, 1.0);
    let ray_direction = normalize(ray_far - ray_origin);
    let texture_origin = (scene.world_to_texture * vec4f(ray_origin, 1.0)).xyz;
    let texture_direction = (
        scene.world_to_texture * vec4f(ray_direction, 0.0)
    ).xyz;
    let inverse_direction = 1.0 / texture_direction;
    let t_bottom = inverse_direction * (scene.texture_min.xyz - texture_origin);
    let t_top = inverse_direction * (scene.texture_max.xyz - texture_origin);
    let t_minimum = min(t_bottom, t_top);
    let t_maximum = max(t_bottom, t_top);
    var t0 = max(0.0, max(t_minimum.x, max(t_minimum.y, t_minimum.z)));
    var t1 = min(t_maximum.x, min(t_maximum.y, t_maximum.z));
    if t0 >= t1 {
        discard;
    }

    let pixel = clamp(
        vec2i(fragment_position.xy),
        vec2i(0),
        vec2i(screen_size) - vec2i(1),
    );
    let opaque_depth = textureLoad(scene_depth, pixel, 0);
    if opaque_depth < 1.0 {
        let opaque_world = world_at(ndc, opaque_depth);
        t1 = min(t1, dot(opaque_world - ray_origin, ray_direction));
    }
    if t0 >= t1 {
        discard;
    }

    let image_mode = scene.plane_positions_mode.w;
    if image_mode > 1.5 && image_mode < 2.5 {
        let slab_normal = scene.tilted_slab_plane.xyz;
        let q0 = dot(slab_normal, texture_origin)
            + scene.tilted_slab_plane.w;
        let q_direction = dot(slab_normal, texture_direction);
        if abs(q_direction) < 1e-12 {
            if q0 < scene.tilted_slab_range.x
                || q0 > scene.tilted_slab_range.y {
                discard;
            }
        } else {
            let slab_t0 = (scene.tilted_slab_range.x - q0) / q_direction;
            let slab_t1 = (scene.tilted_slab_range.y - q0) / q_direction;
            t0 = max(t0, min(slab_t0, slab_t1));
            t1 = min(t1, max(slab_t0, slab_t1));
            if t0 >= t1 {
                discard;
            }
        }
    }

    let step_length = max(scene.screen_step_colormap.z, 0.00005);
    let gradient_offset = min(0.5 * step_length, 0.002);
    var result = vec4f(0.0);

    if image_mode > 0.5 && image_mode < 1.5 {
        var plane_t = array<f32, 3>(1e30, 1e30, 1e30);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            if scene.plane_mask[axis] > 0.5
                && abs(texture_direction[axis]) > 1e-12 {
                let candidate = (
                    scene.plane_positions_mode[axis] - texture_origin[axis]
                ) / texture_direction[axis];
                let point = texture_origin + candidate * texture_direction;
                if candidate >= t0 && candidate <= t1
                    && all(point >= scene.texture_min.xyz - vec3f(1e-5))
                    && all(point <= scene.texture_max.xyz + vec3f(1e-5)) {
                    plane_t[axis] = candidate;
                }
            }
        }
        for (var plane_index = 0u; plane_index < 3u; plane_index += 1u) {
            var nearest_t = 1e30;
            var nearest_axis = 3u;
            for (var axis = 0u; axis < 3u; axis += 1u) {
                if plane_t[axis] < nearest_t {
                    nearest_t = plane_t[axis];
                    nearest_axis = axis;
                }
            }
            if nearest_axis >= 3u {
                break;
            }
            plane_t[nearest_axis] = 1e30;
            let sample_position = clamp(
                texture_origin + nearest_t * texture_direction,
                vec3f(0.001),
                vec3f(0.999),
            );
            let sample_color = sample_color_at(
                sample_position, ray_direction, gradient_offset, step_length
            );
            if scene.render_options.y > 0.5 {
                result = max(result, sample_color);
            } else {
                let contribution = (1.0 - result.a) * sample_color.a;
                result = vec4f(
                    result.rgb + contribution * sample_color.rgb,
                    result.a + contribution,
                );
            }
        }
        if result.a <= 0.0 {
            discard;
        }
        if scene.render_options.y > 0.5 {
            return vec4f(result.rgb * result.a, result.a);
        }
        return result;
    }

    var texcoord = texture_origin + t0 * texture_direction;
    let texcoord_end = texture_origin + t1 * texture_direction;
    let texture_ray = texcoord_end - texcoord;
    var remaining = length(texture_ray);
    let step_vector = step_length * texture_ray / max(remaining, 1e-20);
    texcoord += random_offset(fragment_position.xy) * step_vector;

    for (var sample_index = 0u; sample_index < 8192u; sample_index += 1u) {
        if remaining <= 0.0
            || (scene.render_options.y < 0.5 && result.a >= 0.99) {
            break;
        }
        let sample_position = clamp(texcoord, vec3f(0.001), vec3f(0.999));
        var sample_color = sample_color_at(
            sample_position, ray_direction, gradient_offset, step_length
        );
        if sample_color.a > 0.001 {
            if scene.render_options.y > 0.5 {
                result = max(result, sample_color);
            } else {
                let contribution = (1.0 - result.a) * sample_color.a;
                result = vec4f(
                    result.rgb + contribution * sample_color.rgb,
                    result.a + contribution,
                );
            }
        }
        remaining -= step_length;
        texcoord += step_vector;
    }
    if result.a <= 0.0 {
        discard;
    }
    if scene.render_options.y > 0.5 {
        return vec4f(result.rgb * result.a, result.a);
    }
    return result;
}
"""


@dataclass
class _GpuMeshBatch:
    mesh: MeshData
    vertex_buffer: object
    normal_buffer: object
    index_buffer: object
    instance_buffer: object
    instance_color_buffer: object
    vertex_color_buffer: object
    lighting_buffer: object
    impostor_buffer: object
    secondary_color_buffer: object


@dataclass
class _GpuHighlightBatch:
    highlight: HighlightData
    vertex_buffer: object
    index_buffer: object
    instance_buffer: object


@dataclass
class _GpuVolumeBatch:
    volume: VolumeData
    texture: object
    texture_view: object
    colormap_texture: object
    colormap_view: object
    uniform_buffer: object
    bind_group: object = None


class WgpuMeshRenderer:
    """Small indexed-mesh renderer around wgpu-py.

    ``canvas`` may be either a rendercanvas canvas or ChimeraX's direct native
    Qt canvas.  The latter owns presentation itself and therefore has no
    rendercanvas dependency.
    """

    depth_format = "depth24plus"
    max_multishadows = 64

    def __init__(self, canvas, mesh=None, *, draw_calls=1, use_render_bundle=True,
                 stats_callback=None, adapter=None, device=None):
        try:
            import wgpu
        except ImportError as error:
            raise RuntimeError(
                "The wgpu renderer needs the optional 'wgpu' Python package."
            ) from error

        self.wgpu = wgpu
        self.canvas = canvas
        self.stats_callback = stats_callback
        self.draw_calls = max(1, int(draw_calls))
        self.use_render_bundle = bool(use_render_bundle)
        self._depth_size = None
        self._depth_view = None
        self._render_bundle = None
        self._frame_count = 0
        self._block_start = perf_counter()
        self._encode_seconds = 0.0
        self.mesh = None
        self.meshes = []
        self._gpu_batches = []
        self.highlights = []
        self._gpu_highlights = []
        self.volumes = []
        self._gpu_volumes = []
        self._volume_depth_view = None
        self._highlight_depth_size = None
        self._highlight_depth_view = None
        self._highlight_overlay_bind_group = None
        self._drag_rectangle = None
        self._drag_rectangle_color = (0.0, 1.0, 0.0, 1.0)
        self._highlight_color = (0.0, 1.0, 0.0, 1.0)
        self._highlight_thickness = 1.0
        self.index_count = 0
        self.instance_count = 0
        self.upload_seconds = 0.0
        self._view_projection = np.eye(4, dtype=np.float32)
        self._lighting_uniforms = np.asarray(
            (
                0.577, -0.577, -0.577, 0.0,
                0.8, 0.8, 0.8, 0.0,
                0.3, 0.3, 0.3, 30.0,
                -0.2, -0.2, -0.959, 0.0,
                0.4, 0.4, 0.4, 0.0,
                0.32, 0.32, 0.32, 0.0,
                0.0, 0.0, 10.0, 0.0,
            ),
            dtype=np.float32,
        )
        self._light_view_projection = np.eye(4, dtype=np.float32)
        self._shadow_params = np.zeros(4, dtype=np.float32)
        self._multishadow_count = 0
        self._multishadow_resource_key = None
        self._direct_shadow_dirty = True
        self._multishadow_dirty = True
        self._direct_shadow_state = None
        self._multishadow_state = None
        self.background_color = (0.035, 0.045, 0.065, 1.0)

        self.context = canvas.get_wgpu_context()
        self.adapter = adapter or wgpu.gpu.request_adapter_sync(
            power_preference="high-performance", canvas=self.context
        )
        if self.adapter is None:
            raise RuntimeError("wgpu could not find an adapter for the Qt canvas")
        if device is None:
            adapter_features = set(getattr(self.adapter, "features", ()))
            required_features = (
                [_CONSERVATIVE_DEPTH_FEATURE]
                if _CONSERVATIVE_DEPTH_FEATURE in adapter_features else []
            )
            self.device = self.adapter.request_device_sync(
                required_features=required_features
            )
        else:
            self.device = device
        self._conservative_depth = _CONSERVATIVE_DEPTH_FEATURE in set(
            getattr(self.device, "features", ())
        )
        preferred_format = self.context.get_preferred_format(self.device.adapter)
        # ChimeraX's OpenGL framebuffer writes linear shader values directly
        # into an unorm surface.  The preferred WebGPU format is normally the
        # sRGB variant, whose implicit encoding made the same colors visibly
        # brighter and less saturated.
        self.color_format = preferred_format.removesuffix("-srgb")
        self.context.configure(device=self.device, format=self.color_format)

        self._create_pipeline()
        if mesh is not None:
            self.upload_mesh(mesh)
        else:
            self._write_uniforms()

    @property
    def adapter_summary(self):
        summary = getattr(self.adapter, "summary", None)
        return summary() if callable(summary) else str(self.adapter.info)

    @property
    def presentation_mode(self):
        if getattr(self.canvas, "manual_present", False):
            return "direct native screen"
        direct = getattr(self.canvas, "_present_to_screen", False)
        return "direct screen" if direct else "bitmap readback"

    def _create_pipeline(self):
        wgpu = self.wgpu
        device = self.device
        self._shader = device.create_shader_module(code=_MESH_SHADER)
        self._impostor_vertex_shaders = {}
        self._impostor_fragment_shaders = {}
        for kind in ("sphere", "cylinder"):
            self._impostor_vertex_shaders[kind] = device.create_shader_module(
                label="%s impostor vertex" % kind,
                code=_IMPOSTOR_VERTEX_GLSL[kind],
            )
            self._impostor_fragment_shaders[kind] = device.create_shader_module(
                label="%s impostor fragment" % kind,
                code=_glsl_fragment_source(
                    _IMPOSTOR_FRAGMENT_GLSL[kind], self._conservative_depth
                ),
            )
        self.uniform_buffer = device.create_buffer(
            size=336,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._multishadow_uniform_buffer = device.create_buffer(
            size=64 * 64 + 64 * 16 + 16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._create_shadow_pipeline()
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "comparison"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "depth",
                        "view_dimension": "2d-array",
                    },
                },
            ]
        )
        self._pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._create_scene_bind_group()
        self._pipelines = {}
        self._impostor_pipelines = {}
        self.pipeline = self._pipeline("triangle-list", "opaque", False)
        self._create_silhouette_pipeline()
        self._create_highlight_pipeline()
        self._create_volume_pipeline()

    def _create_shadow_pipeline(self):
        wgpu = self.wgpu
        device = self.device
        self._shadow_size = None
        self._ensure_shadow_map_size(2048)
        self._shadow_sampler = device.create_sampler(
            compare="less-equal",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            min_filter="linear",
            mag_filter="linear",
        )
        self._shadow_uniform_buffer = device.create_buffer(
            size=144,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._shadow_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        self._shadow_bind_group = device.create_bind_group(
            layout=self._shadow_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._shadow_uniform_buffer,
                        "offset": 0,
                        "size": 144,
                    },
                }
            ],
        )
        shader = device.create_shader_module(code=_SHADOW_SHADER)
        self._shadow_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._shadow_layout]),
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 12,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    },
                    {
                        "array_stride": 64,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x4", "offset": 0, "shader_location": 2},
                            {"format": "float32x4", "offset": 16, "shader_location": 3},
                            {"format": "float32x4", "offset": 32, "shader_location": 4},
                            {"format": "float32x4", "offset": 48, "shader_location": 5},
                        ],
                    },
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": True,
                "depth_compare": "less",
                "depth_bias": 2,
                "depth_bias_slope_scale": 1.5,
                "depth_bias_clamp": 0.0,
            },
        )
        self._impostor_shadow_vertex_shaders = {}
        self._impostor_shadow_fragment_shaders = {}
        for kind in ("sphere", "cylinder"):
            self._impostor_shadow_vertex_shaders[kind] = (
                device.create_shader_module(
                    label="%s impostor shadow vertex" % kind,
                    code=_IMPOSTOR_SHADOW_VERTEX_GLSL[kind],
                )
            )
            self._impostor_shadow_fragment_shaders[kind] = (
                device.create_shader_module(
                    label="%s impostor shadow fragment" % kind,
                    code=_glsl_fragment_source(
                        _IMPOSTOR_SHADOW_FRAGMENT_GLSL[kind],
                        self._conservative_depth,
                    ),
                )
            )
        self._impostor_shadow_pipelines = {}
        self._multishadow_matrix_buffer = device.create_buffer(
            size=self.max_multishadows * 256,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._ensure_multishadow_resources(0, 1)

    def _ensure_shadow_map_size(self, size):
        size = max(1, int(size))
        if size == self._shadow_size:
            return
        self._shadow_size = size
        self._shadow_texture = self.device.create_texture(
            size=(size, size, 1),
            format="depth32float",
            usage=(
                self.wgpu.TextureUsage.RENDER_ATTACHMENT
                | self.wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._shadow_view = self._shadow_texture.create_view()
        if hasattr(self, "_bind_group_layout"):
            self._create_scene_bind_group()
            self._render_bundle = None
        self._direct_shadow_dirty = True

    def _ensure_multishadow_resources(self, count, map_size):
        """Create a depth-array layer per ambient-shadow direction."""

        from math import ceil, sqrt

        requested_count = max(0, int(count))
        layer_count = max(1, requested_count)
        # OpenGL tiles all maps into map_size squared.  Equal-sized array
        # layers preserve its effective resolution and avoid atlas seams.
        layer_size = (
            max(1, int(map_size) // int(ceil(sqrt(layer_count))))
            if requested_count
            else 1
        )
        key = layer_count, layer_size
        if key == self._multishadow_resource_key:
            return
        wgpu = self.wgpu
        self._multishadow_texture = self.device.create_texture(
            size=(layer_size, layer_size, layer_count),
            format="depth32float",
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING),
        )
        self._multishadow_sample_view = self._multishadow_texture.create_view(
            dimension="2d-array",
            base_array_layer=0,
            array_layer_count=layer_count,
        )
        self._multishadow_layer_views = [
            self._multishadow_texture.create_view(
                dimension="2d", base_array_layer=i, array_layer_count=1
            )
            for i in range(layer_count)
        ]
        self._multishadow_bind_groups = [
            self.device.create_bind_group(
                layout=self._shadow_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._multishadow_matrix_buffer,
                            "offset": i * 256,
                            "size": 144,
                        },
                    }
                ],
            )
            for i in range(layer_count)
        ]
        self._multishadow_resource_key = key
        self._multishadow_dirty = True
        if hasattr(self, "_bind_group_layout"):
            self._create_scene_bind_group()
            self._render_bundle = None

    def _create_scene_bind_group(self):
        self.bind_group = self.device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": 336,
                    },
                },
                {"binding": 1, "resource": self._shadow_view},
                {"binding": 2, "resource": self._shadow_sampler},
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self._multishadow_uniform_buffer,
                        "offset": 0,
                        "size": 64 * 64 + 64 * 16 + 16,
                    },
                },
                {"binding": 4, "resource": self._multishadow_sample_view},
            ],
        )

    def _create_silhouette_pipeline(self):
        wgpu = self.wgpu
        device = self.device
        self._silhouette_enabled = False
        self._silhouette_uniform_buffer = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._silhouette_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "depth",
                        "view_dimension": "2d",
                    },
                },
            ]
        )
        shader = device.create_shader_module(code=_SILHOUETTE_SHADER)
        self._silhouette_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self._silhouette_layout]
            ),
            vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.color_format,
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                    }
                ],
            },
        )
        self._silhouette_bind_group = None
        self._silhouette_depth_view = None

    def _create_highlight_pipeline(self):
        device = self.device
        wgpu = self.wgpu
        depth_shader = device.create_shader_module(code=_HIGHLIGHT_DEPTH_SHADER)
        self._highlight_depth_pipeline = device.create_render_pipeline(
            layout=self._pipeline_layout,
            vertex={
                "module": depth_shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 12,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    },
                    {
                        "array_stride": 64,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x4", "offset": 0, "shader_location": 2},
                            {"format": "float32x4", "offset": 16, "shader_location": 3},
                            {"format": "float32x4", "offset": 32, "shader_location": 4},
                            {"format": "float32x4", "offset": 48, "shader_location": 5},
                        ],
                    },
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": self.depth_format,
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
        )
        self._highlight_overlay_uniform_buffer = device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._highlight_overlay_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
            ]
        )
        overlay_shader = device.create_shader_module(code=_HIGHLIGHT_OVERLAY_SHADER)
        self._highlight_overlay_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self._highlight_overlay_layout]
            ),
            vertex={"module": overlay_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            fragment={
                "module": overlay_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.color_format}],
            },
        )

    def _create_volume_pipeline(self):
        wgpu = self.wgpu
        self._volume_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "3d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "1d"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
            ]
        )
        shader = self.device.create_shader_module(code=_VOLUME_SHADER)
        self._volume_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._volume_layout]
            ),
            vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.color_format,
                        # The ray marcher returns premultiplied RGB.
                        "blend": {
                            "color": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                    }
                ],
            },
        )
        sampler_options = {
            "address_mode_u": "clamp-to-edge",
            "address_mode_v": "clamp-to-edge",
            "address_mode_w": "clamp-to-edge",
        }
        self._volume_samplers = {
            True: self.device.create_sampler(
                min_filter="linear", mag_filter="linear", **sampler_options
            ),
            False: self.device.create_sampler(
                min_filter="nearest", mag_filter="nearest", **sampler_options
            ),
        }
        self._colormap_sampler = self._volume_samplers[True]

    def _pipeline(self, topology, render_pass, vertex_colors_are_instance=False):
        key = topology, render_pass, vertex_colors_are_instance
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline

        wgpu = self.wgpu
        transparent = render_pass in ("transparent", "last")
        on_top = render_pass == "last"
        target = {"format": self.color_format}
        if transparent:
            target["blend"] = {
                "color": {
                    "src_factor": "src-alpha",
                    "dst_factor": "one-minus-src-alpha",
                    "operation": "add",
                },
                "alpha": {
                    "src_factor": "one",
                    "dst_factor": "one-minus-src-alpha",
                    "operation": "add",
                },
            }
        pipeline = self.device.create_render_pipeline(
            layout=self._pipeline_layout,
            vertex={
                "module": self._shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 12,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    },
                    {
                        "array_stride": 12,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 1}
                        ],
                    },
                    {
                        "array_stride": 64,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x4", "offset": 0, "shader_location": 2},
                            {"format": "float32x4", "offset": 16, "shader_location": 3},
                            {"format": "float32x4", "offset": 32, "shader_location": 4},
                            {"format": "float32x4", "offset": 48, "shader_location": 5},
                        ],
                    },
                    {
                        "array_stride": 4,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "unorm8x4", "offset": 0, "shader_location": 6}
                        ],
                    },
                    {
                        "array_stride": 4,
                        "step_mode": (
                            "instance" if vertex_colors_are_instance else "vertex"
                        ),
                        "attributes": [
                            {"format": "unorm8x4", "offset": 0, "shader_location": 7}
                        ],
                    },
                    {
                        "array_stride": 12,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 8}
                        ],
                    },
                ],
            },
            primitive={
                "topology": topology,
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": self.depth_format,
                "depth_write_enabled": not transparent and not on_top,
                "depth_compare": "always" if on_top else "less",
            },
            multisample=None,
            fragment={
                "module": self._shader,
                "entry_point": "fs_main",
                "targets": [target],
            },
        )
        self._pipelines[key] = pipeline
        return pipeline

    def _impostor_pipeline(self, kind, render_pass):
        key = kind, render_pass
        pipeline = self._impostor_pipelines.get(key)
        if pipeline is not None:
            return pipeline

        transparent = render_pass in ("transparent", "last")
        on_top = render_pass == "last"
        target = {"format": self.color_format}
        if transparent:
            target["blend"] = {
                "color": {
                    "src_factor": "src-alpha",
                    "dst_factor": "one-minus-src-alpha",
                    "operation": "add",
                },
                "alpha": {
                    "src_factor": "one",
                    "dst_factor": "one-minus-src-alpha",
                    "operation": "add",
                },
            }

        position_buffer = {
            "array_stride": 12,
            "step_mode": "vertex",
            "attributes": [
                {"format": "float32x3", "offset": 0, "shader_location": 0}
            ],
        }
        color_buffer = {
            "array_stride": 4,
            "step_mode": "instance",
            "attributes": [
                {"format": "unorm8x4", "offset": 0, "shader_location": 6}
            ],
        }
        options_buffer = {
            "array_stride": 16,
            "step_mode": "instance",
            "attributes": [
                {"format": "float32x4", "offset": 0, "shader_location": 8}
            ],
        }
        if kind == "sphere":
            buffers = [
                position_buffer,
                {
                    "array_stride": 16,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "float32x4", "offset": 0, "shader_location": 9}
                    ],
                },
                color_buffer,
                options_buffer,
            ]
        else:
            buffers = [
                position_buffer,
                {
                    "array_stride": 64,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "float32x4", "offset": 0, "shader_location": 2},
                        {"format": "float32x4", "offset": 16, "shader_location": 3},
                        {"format": "float32x4", "offset": 32, "shader_location": 4},
                        {"format": "float32x4", "offset": 48, "shader_location": 5},
                    ],
                },
                color_buffer,
                {
                    "array_stride": 4,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "unorm8x4", "offset": 0, "shader_location": 9}
                    ],
                },
                options_buffer,
            ]
        pipeline = self.device.create_render_pipeline(
            layout=self._pipeline_layout,
            vertex={
                "module": self._impostor_vertex_shaders[kind],
                "entry_point": "main",
                "buffers": buffers,
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": self.depth_format,
                "depth_write_enabled": not transparent and not on_top,
                "depth_compare": "always" if on_top else "less",
            },
            fragment={
                "module": self._impostor_fragment_shaders[kind],
                "entry_point": "main",
                "targets": [target],
            },
        )
        self._impostor_pipelines[key] = pipeline
        return pipeline

    def _impostor_shadow_pipeline(self, kind):
        pipeline = self._impostor_shadow_pipelines.get(kind)
        if pipeline is not None:
            return pipeline

        position_buffer = {
            "array_stride": 12,
            "step_mode": "vertex",
            "attributes": [
                {"format": "float32x3", "offset": 0, "shader_location": 0}
            ],
        }
        if kind == "sphere":
            buffers = [
                position_buffer,
                {
                    "array_stride": 16,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "float32x4", "offset": 0, "shader_location": 9}
                    ],
                },
            ]
        else:
            buffers = [
                position_buffer,
                {
                    "array_stride": 64,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "float32x4", "offset": 0, "shader_location": 2},
                        {"format": "float32x4", "offset": 16, "shader_location": 3},
                        {"format": "float32x4", "offset": 32, "shader_location": 4},
                        {"format": "float32x4", "offset": 48, "shader_location": 5},
                    ],
                },
                {
                    "array_stride": 16,
                    "step_mode": "instance",
                    "attributes": [
                        {"format": "float32x4", "offset": 0, "shader_location": 8}
                    ],
                },
            ]
        pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._shadow_layout]
            ),
            vertex={
                "module": self._impostor_shadow_vertex_shaders[kind],
                "entry_point": "main",
                "buffers": buffers,
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": True,
                "depth_compare": "less",
                # The proxy is at zero depth, which is the lower bound promised
                # by GLSL depth_greater.  Fixed-function biasing of that proxy
                # would invalidate the promise; receiver bias is applied while
                # sampling the resulting shadow map instead.
                "depth_bias": 0,
                "depth_bias_slope_scale": 0.0,
                "depth_bias_clamp": 0.0,
            },
            fragment={
                "module": self._impostor_shadow_fragment_shaders[kind],
                "entry_point": "main",
                "targets": [],
            },
        )
        self._impostor_shadow_pipelines[kind] = pipeline
        return pipeline

    def upload_mesh(self, mesh):
        """Upload NumPy arrays through their buffer-protocol views."""

        self.upload_meshes([mesh])
        self._fit_transform = _fit_transform(mesh.bounds)
        self._view_projection = self._fit_transform
        self._write_uniforms()

    def upload_meshes(self, meshes):
        """Replace the rendered scene with ordered immutable mesh batches."""

        wgpu = self.wgpu
        meshes = list(meshes)
        start = perf_counter()
        batches = []
        for mesh in meshes:
            options = [
                1.0 if mesh.use_lighting else 0.0,
                1.0 if mesh.accept_shadow else 0.0,
                1.0 if mesh.accept_multishadow else 0.0,
            ]
            if mesh.impostor_kind:
                options.append(float(mesh.impostor_dashes))
            drawing_options = np.tile(
                np.asarray(options, dtype=np.float32),
                (mesh.instance_count, 1),
            )
            impostor_buffer = (
                self.device.create_buffer_with_data(
                    data=mesh.impostor_data, usage=wgpu.BufferUsage.VERTEX
                )
                if mesh.impostor_kind else None
            )
            secondary_color_buffer = (
                self.device.create_buffer_with_data(
                    data=mesh.secondary_colors, usage=wgpu.BufferUsage.VERTEX
                )
                if mesh.impostor_kind == "cylinder" else None
            )
            batch = _GpuMeshBatch(
                mesh,
                self.device.create_buffer_with_data(
                    data=mesh.vertices, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=mesh.normals, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=mesh.triangles, usage=wgpu.BufferUsage.INDEX
                ),
                self.device.create_buffer_with_data(
                    data=mesh.instance_matrices, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=mesh.instance_colors, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=mesh.vertex_colors, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=drawing_options, usage=wgpu.BufferUsage.VERTEX
                ),
                impostor_buffer,
                secondary_color_buffer,
            )
            batches.append(batch)
        self.upload_seconds = perf_counter() - start
        self.meshes = meshes
        self._gpu_batches = batches
        self.mesh = self.meshes[0] if len(self.meshes) == 1 else None
        self.index_count = sum(mesh.triangles.size for mesh in self.meshes)
        self.instance_count = sum(mesh.instance_count for mesh in self.meshes)
        self._render_bundle = None
        self._direct_shadow_dirty = True
        self._multishadow_dirty = True
        self._reset_stats()

    def upload_highlights(self, highlights):
        """Replace the selected-geometry batches used by the outline pass."""

        wgpu = self.wgpu
        highlights = list(highlights)
        self._gpu_highlights = [
            _GpuHighlightBatch(
                highlight,
                self.device.create_buffer_with_data(
                    data=highlight.vertices, usage=wgpu.BufferUsage.VERTEX
                ),
                self.device.create_buffer_with_data(
                    data=highlight.triangles, usage=wgpu.BufferUsage.INDEX
                ),
                self.device.create_buffer_with_data(
                    data=highlight.instance_matrices, usage=wgpu.BufferUsage.VERTEX
                ),
            )
            for highlight in highlights
        ]
        self.highlights = highlights

    def set_highlight(self, color, thickness):
        self._highlight_color = tuple(float(component) for component in color)
        self._highlight_thickness = max(1.0, float(thickness))

    def set_drag_rectangle(self, rectangle, color=None):
        """Set a top-left-origin physical-pixel rectangle, or clear it."""

        self._drag_rectangle = rectangle
        if color is not None:
            self._drag_rectangle_color = tuple(
                float(component) / 255.0 for component in color
            )

    def _upload_sampled_texture(self, values, dimension):
        """Upload a contiguous NumPy texture through the buffer protocol."""

        values = np.asarray(values)
        component_count = values.shape[dimension] if values.ndim > dimension else 1
        if component_count == 3:
            padded = np.empty(values.shape[:-1] + (4,), dtype=values.dtype)
            padded[..., :3] = values
            if np.issubdtype(values.dtype, np.integer):
                padded[..., 3] = np.iinfo(values.dtype).max
            else:
                padded[..., 3] = 1
            values = padded
            component_count = 4
        if component_count not in (1, 2, 4):
            raise ValueError("WGPU textures need 1, 2, or 4 components")
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)

        prefixes = {
            np.dtype(np.uint8): "8unorm",
            np.dtype(np.int8): "8snorm",
            np.dtype(np.float16): "16float",
        }
        if values.dtype == np.uint16:
            values = (values.astype(np.float32) / 65535).astype(np.float16)
        elif values.dtype == np.int16:
            values = np.maximum(
                values.astype(np.float32) / 32767, -1
            ).astype(np.float16)
        elif values.dtype == np.float32:
            # Filterable float32 textures are an optional WebGPU feature;
            # float16 is core and preserves the map's dynamic range here.
            values = values.astype(np.float16)
        suffix = prefixes.get(values.dtype)
        if suffix is None:
            raise ValueError("unsupported WGPU volume dtype %s" % values.dtype)
        channel_prefix = {1: "r", 2: "rg", 4: "rgba"}[component_count]
        texture_format = channel_prefix + suffix

        if dimension == 3:
            depth, height, width = values.shape[:3]
            size = (width, height, depth)
            rows_per_image = height
        else:
            width = values.shape[0]
            size = (width, 1, 1)
            rows_per_image = 1
        bytes_per_texel = component_count * values.dtype.itemsize
        texture = self.device.create_texture(
            size=size,
            dimension="3d" if dimension == 3 else "1d",
            format=texture_format,
            usage=(
                self.wgpu.TextureUsage.TEXTURE_BINDING
                | self.wgpu.TextureUsage.COPY_DST
            ),
        )
        self.device.queue.write_texture(
            {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
            values,
            {
                "offset": 0,
                "bytes_per_row": width * bytes_per_texel,
                "rows_per_image": rows_per_image,
            },
            size,
        )
        return texture, texture.create_view()

    def upload_volumes(self, volumes):
        """Upload currently displayed 3-D image textures."""

        texture_cache = {
            (
                id(batch.volume.values),
                batch.volume.values.shape,
                batch.volume.values.dtype.str,
            ): (batch.texture, batch.texture_view)
            for batch in self._gpu_volumes
        }
        colormap_cache = {
            (
                id(batch.volume.colormap),
                batch.volume.colormap.shape,
                batch.volume.colormap.dtype.str,
            ): (batch.colormap_texture, batch.colormap_view)
            for batch in self._gpu_volumes
            if batch.volume.colormap is not None
        }
        for batch in self._gpu_volumes:
            if batch.volume.colormap is None:
                colormap_cache[("dummy",)] = (
                    batch.colormap_texture,
                    batch.colormap_view,
                )
                break
        dummy_colormap = np.asarray(
            ((0, 0, 0, 0), (255, 255, 255, 255)), np.uint8
        )
        gpu_volumes = []
        for volume in volumes:
            try:
                texture_key = (
                    id(volume.values), volume.values.shape, volume.values.dtype.str
                )
                uploaded = texture_cache.get(texture_key)
                if uploaded is None:
                    uploaded = self._upload_sampled_texture(volume.values, 3)
                    texture_cache[texture_key] = uploaded
                texture, texture_view = uploaded
                colormap = volume.colormap
                if colormap is None:
                    colormap = dummy_colormap
                    colormap_key = ("dummy",)
                else:
                    colormap_key = (
                        id(colormap), colormap.shape, colormap.dtype.str
                    )
                uploaded_colormap = colormap_cache.get(colormap_key)
                if uploaded_colormap is None:
                    uploaded_colormap = self._upload_sampled_texture(colormap, 1)
                    colormap_cache[colormap_key] = uploaded_colormap
                colormap_texture, colormap_view = uploaded_colormap
            except ValueError:
                continue
            uniform_buffer = self.device.create_buffer(
                size=384,
                usage=(
                    self.wgpu.BufferUsage.UNIFORM
                    | self.wgpu.BufferUsage.COPY_DST
                ),
            )
            gpu_volumes.append(
                _GpuVolumeBatch(
                    volume,
                    texture,
                    texture_view,
                    colormap_texture,
                    colormap_view,
                    uniform_buffer,
                )
            )
        self.volumes = list(volumes)
        self._gpu_volumes = gpu_volumes
        self._volume_depth_view = None

    def clear_mesh(self):
        """Draw only the background until another mesh is uploaded."""

        self.mesh = None
        self.meshes = []
        self._gpu_batches = []
        self.index_count = 0
        self.instance_count = 0
        self._render_bundle = None
        self._direct_shadow_dirty = True
        self._multishadow_dirty = True

    def set_view_projection(self, matrix):
        """Set a conventional row-major scene-to-WebGPU-clip matrix."""

        self._view_projection = np.asarray(matrix, dtype=np.float32)

    def set_background_color(self, rgba):
        self.background_color = tuple(float(c) for c in rgba)

    def set_lighting(self, view):
        """Copy ChimeraX lighting/material state into the WGPU scene block."""

        lighting = view.lighting
        material = view.material
        self._ensure_shadow_map_size(lighting.shadow_map_size)

        def direction(values):
            vector = np.asarray(values, dtype=np.float32)
            length = float(np.linalg.norm(vector))
            if length:
                vector = vector / length
            if lighting.move_lights_with_camera:
                vector = view.camera.position.transform_vector(vector)
            return tuple(vector) + (0.0,)

        def color(values, scale):
            return tuple(float(scale) * float(c) for c in values) + (0.0,)

        key_direction = direction(lighting.key_light_direction)
        fill_direction = direction(lighting.fill_light_direction)
        camera_position = tuple(
            float(c) for c in view.camera.get_position(0).origin()
        ) + (0.0,)
        specular_color = color(
            lighting.key_light_color,
            material.specular_reflectivity * lighting.key_light_intensity,
        )
        uniforms = (
            *key_direction,
            *color(
                lighting.key_light_color,
                material.diffuse_reflectivity * lighting.key_light_intensity,
            ),
            *specular_color[:3],
            float(material.specular_exponent),
            *fill_direction,
            *color(
                lighting.fill_light_color,
                material.diffuse_reflectivity * lighting.fill_light_intensity,
            ),
            *color(
                lighting.ambient_light_color,
                material.ambient_reflectivity * lighting.ambient_light_intensity,
            ),
            *camera_position,
        )
        self._lighting_uniforms = np.asarray(uniforms, dtype=np.float32)
        self._light_view_projection = _shadow_view_projection(
            self.meshes, key_direction[:3]
        )
        self._shadow_params[:] = (
            1.0 if lighting.shadows and self.meshes else 0.0,
            float(lighting.shadow_depth_bias),
            0.0,
            0.0,
        )

        count = min(self.max_multishadows, max(0, int(lighting.multishadow)))
        self._ensure_multishadow_resources(count, lighting.multishadow_map_size)
        self._multishadow_count = count if self.meshes else 0
        directions = _multishadow_directions(self._multishadow_count)
        matrices = np.zeros(
            (self.max_multishadows, 4, 4), dtype=np.float32
        )
        direction_vectors = np.zeros(
            (self.max_multishadows, 4), dtype=np.float32
        )
        for i, multi_direction in enumerate(directions):
            matrices[i] = _shadow_view_projection(self.meshes, multi_direction)
            direction_vectors[i, :3] = multi_direction

        # WGSL matrices are column-major.
        matrix_bytes = np.ascontiguousarray(matrices.transpose(0, 2, 1))
        multishadow_data = np.concatenate(
            (
                matrix_bytes.reshape(-1),
                direction_vectors.reshape(-1),
                np.asarray(
                    (
                        float(self._multishadow_count),
                        float(lighting.multishadow_depth_bias),
                        0.0,
                        0.0,
                    ),
                    dtype=np.float32,
                ),
            )
        )
        self.device.queue.write_buffer(
            self._multishadow_uniform_buffer, 0, multishadow_data
        )

        # WebGPU uniform-buffer offsets are 256-byte aligned.  Each light gets
        # its transform, inverse transform, and viewport in one padded slot so
        # analytic impostors can reconstruct light rays in the depth pass.
        padded_matrices = np.zeros(
            (self.max_multishadows, 64), dtype=np.float32
        )
        if self._multishadow_count:
            layer_size = float(self._multishadow_resource_key[1])
            for index in range(self._multishadow_count):
                padded_matrices[index, :36] = _shadow_ray_uniforms(
                    matrices[index], (layer_size, layer_size)
                )
        self.device.queue.write_buffer(
            self._multishadow_matrix_buffer, 0, padded_matrices
        )

        shadow_uniforms = _shadow_ray_uniforms(
            self._light_view_projection,
            (float(self._shadow_size), float(self._shadow_size)),
        )
        self.device.queue.write_buffer(
            self._shadow_uniform_buffer, 0, shadow_uniforms
        )
        direct_shadow_state = (
            bool(self._shadow_params[0] > 0.5),
            shadow_uniforms.tobytes() if self._shadow_params[0] > 0.5 else b"",
            self._shadow_params.tobytes(),
            self._shadow_size,
        )
        if direct_shadow_state != self._direct_shadow_state:
            self._direct_shadow_dirty = True
            self._direct_shadow_state = direct_shadow_state
        multishadow_state = (
            matrix_bytes[:self._multishadow_count].tobytes(),
            float(lighting.multishadow_depth_bias),
            self._multishadow_resource_key,
            len(self._gpu_batches),
        )
        if multishadow_state != self._multishadow_state:
            self._multishadow_dirty = True
            self._multishadow_state = multishadow_state
        self._write_uniforms()

    def set_silhouette(self, silhouette, size):
        self._silhouette_enabled = bool(silhouette.enabled)
        if not self._silhouette_enabled:
            return
        color = tuple(float(c) for c in silhouette.color)
        values = np.asarray(
            (
                *color,
                float(size[0]),
                float(size[1]),
                float(silhouette.thickness),
                float(silhouette.depth_jump),
            ),
            dtype=np.float32,
        )
        self.device.queue.write_buffer(self._silhouette_uniform_buffer, 0, values)

    def set_draw_calls(self, draw_calls):
        draw_calls = max(1, int(draw_calls))
        if draw_calls != self.draw_calls:
            self.draw_calls = draw_calls
            self._render_bundle = None
            self._reset_stats()

    def set_use_render_bundle(self, use_render_bundle):
        use_render_bundle = bool(use_render_bundle)
        if use_render_bundle != self.use_render_bundle:
            self.use_render_bundle = use_render_bundle
            self._reset_stats()

    def _reset_stats(self):
        self._frame_count = 0
        self._encode_seconds = 0.0
        self._block_start = perf_counter()

    def _write_uniforms(self):
        # WGSL matrices are column-major.  Transposing a conventional NumPy
        # row-major matrix produces the byte order expected by mat4x4f.
        matrix = np.ascontiguousarray(self._view_projection.T, dtype=np.float32)
        light_matrix = np.ascontiguousarray(
            self._light_view_projection.T, dtype=np.float32
        )
        inverse_matrix = np.ascontiguousarray(
            np.linalg.inv(self._view_projection).T, dtype=np.float32
        )
        width, height = self.canvas.get_physical_size()
        viewport = np.asarray(
            (max(1, width), max(1, height), 0.0, 0.0), dtype=np.float32
        )
        uniform_data = np.concatenate(
            (
                matrix.reshape(-1),
                light_matrix.reshape(-1),
                self._lighting_uniforms,
                self._shadow_params,
                inverse_matrix.reshape(-1),
                viewport,
            )
        )
        self.device.queue.write_buffer(self.uniform_buffer, 0, uniform_data)

    def _ensure_depth_view(self, size):
        if size == self._depth_size:
            return self._depth_view
        width, height = size
        texture = self.device.create_texture(
            size=(width, height, 1),
            format=self.depth_format,
            usage=(
                self.wgpu.TextureUsage.RENDER_ATTACHMENT
                | self.wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._depth_texture = texture
        self._depth_view = texture.create_view()
        self._depth_size = size
        self._silhouette_bind_group = None
        self._volume_depth_view = None
        self._highlight_overlay_bind_group = None
        return self._depth_view

    def _ensure_highlight_depth_view(self, size):
        if size == self._highlight_depth_size:
            return self._highlight_depth_view
        width, height = size
        texture = self.device.create_texture(
            size=(width, height, 1),
            format=self.depth_format,
            usage=(
                self.wgpu.TextureUsage.RENDER_ATTACHMENT
                | self.wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self._highlight_depth_texture = texture
        self._highlight_depth_view = texture.create_view()
        self._highlight_depth_size = size
        self._highlight_overlay_bind_group = None
        return self._highlight_depth_view

    def _update_volume_bind_groups(self):
        if self._volume_depth_view is self._depth_view:
            return
        for batch in self._gpu_volumes:
            batch.bind_group = self.device.create_bind_group(
                layout=self._volume_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": batch.uniform_buffer,
                            "offset": 0,
                            "size": 384,
                        },
                    },
                    {"binding": 1, "resource": batch.texture_view},
                    {
                        "binding": 2,
                        "resource": self._volume_samplers[
                            batch.volume.linear_interpolation
                        ],
                    },
                    {"binding": 3, "resource": batch.colormap_view},
                    {"binding": 4, "resource": self._colormap_sampler},
                    {"binding": 5, "resource": self._depth_view},
                ],
            )
        self._volume_depth_view = self._depth_view

    def _draw_volumes(self, command_encoder, target_view, size):
        if not self._gpu_volumes:
            return
        self._update_volume_bind_groups()
        inverse_view_projection = np.ascontiguousarray(
            np.linalg.inv(self._view_projection).T, dtype=np.float32
        )
        for batch in self._gpu_volumes:
            volume = batch.volume
            lower, upper = volume.texture_bounds
            world_to_texture = np.ascontiguousarray(
                volume.world_to_texture.T, dtype=np.float32
            )
            volume_uniforms = np.concatenate(
                (
                    inverse_view_projection.reshape(-1),
                    world_to_texture.reshape(-1),
                    self._lighting_uniforms[:24],
                    np.asarray((*lower, 0.0), dtype=np.float32),
                    np.asarray((*upper, 0.0), dtype=np.float32),
                    np.asarray(
                        (
                            float(size[0]),
                            float(size[1]),
                            volume.ray_step,
                            1.0 if volume.colormap is not None else 0.0,
                        ),
                        dtype=np.float32,
                    ),
                    np.asarray(
                        (
                            *volume.colormap_range,
                            float(volume.colormap_component_count),
                            0.0,
                        ),
                        dtype=np.float32,
                    ),
                    np.asarray(volume.color, dtype=np.float32),
                    np.asarray(
                        (
                            1.0 if volume.raycast_lighting else 0.0,
                            (
                                1.0
                                if volume.maximum_intensity_projection
                                else 0.0
                            ),
                            1.0 if volume.alpha_correction else 0.0,
                            float(volume.component_count),
                        ),
                        dtype=np.float32,
                    ),
                    np.asarray(
                        (
                            *volume.orthoplane_positions,
                            {
                                "full region": 0,
                                "orthoplanes": 1,
                                "tilted slab": 2,
                                "box faces": 3,
                            }.get(volume.image_mode, 0),
                        ),
                        dtype=np.float32,
                    ),
                    np.asarray(
                        (*volume.orthoplanes_shown, 0.0), dtype=np.float32
                    ),
                    np.asarray(volume.tilted_slab_plane, dtype=np.float32),
                    np.asarray(
                        (*volume.tilted_slab_range, 0.0, 0.0), dtype=np.float32
                    ),
                )
            )
            self.device.queue.write_buffer(
                batch.uniform_buffer, 0, volume_uniforms
            )
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "resolve_target": None,
                    "load_op": self.wgpu.LoadOp.load,
                    "store_op": self.wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.set_pipeline(self._volume_pipeline)
        for batch in self._gpu_volumes:
            render_pass.set_bind_group(0, batch.bind_group)
            render_pass.draw(3)
        render_pass.end()

    def _draw_silhouette(self, command_encoder, target_view):
        if not self._silhouette_enabled:
            return
        if self._silhouette_bind_group is None:
            self._silhouette_bind_group = self.device.create_bind_group(
                layout=self._silhouette_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._silhouette_uniform_buffer,
                            "offset": 0,
                            "size": 32,
                        },
                    },
                    {"binding": 1, "resource": self._depth_view},
                ],
            )
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "resolve_target": None,
                    "load_op": self.wgpu.LoadOp.load,
                    "store_op": self.wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.set_pipeline(self._silhouette_pipeline)
        render_pass.set_bind_group(0, self._silhouette_bind_group)
        render_pass.draw(3)
        render_pass.end()

    def _draw_highlight_overlay(self, command_encoder, target_view, size):
        if not self._gpu_highlights and self._drag_rectangle is None:
            return

        highlight_depth_view = self._ensure_highlight_depth_view(size)
        depth_pass = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": highlight_depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": self.wgpu.LoadOp.clear,
                "depth_store_op": self.wgpu.StoreOp.store,
            },
        )
        if self._gpu_highlights:
            depth_pass.set_pipeline(self._highlight_depth_pipeline)
            depth_pass.set_bind_group(0, self.bind_group)
            for batch in self._gpu_highlights:
                depth_pass.set_vertex_buffer(0, batch.vertex_buffer)
                depth_pass.set_vertex_buffer(1, batch.instance_buffer)
                depth_pass.set_index_buffer(batch.index_buffer, "uint32")
                depth_pass.draw_indexed(
                    batch.highlight.triangles.size,
                    len(batch.highlight.instance_matrices),
                )
        depth_pass.end()

        rectangle = self._drag_rectangle or (0.0, 0.0, 0.0, 0.0)
        overlay_uniforms = np.asarray(
            (
                *self._highlight_color,
                *self._drag_rectangle_color,
                float(size[0]),
                float(size[1]),
                self._highlight_thickness,
                1.0 if self._drag_rectangle is not None else 0.0,
                *rectangle,
            ),
            dtype=np.float32,
        )
        self.device.queue.write_buffer(
            self._highlight_overlay_uniform_buffer, 0, overlay_uniforms
        )
        if self._highlight_overlay_bind_group is None:
            self._highlight_overlay_bind_group = self.device.create_bind_group(
                layout=self._highlight_overlay_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._highlight_overlay_uniform_buffer,
                            "offset": 0,
                            "size": 64,
                        },
                    },
                    {"binding": 1, "resource": highlight_depth_view},
                    {"binding": 2, "resource": self._depth_view},
                ],
            )
        overlay_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "resolve_target": None,
                    "load_op": self.wgpu.LoadOp.load,
                    "store_op": self.wgpu.StoreOp.store,
                }
            ]
        )
        overlay_pass.set_pipeline(self._highlight_overlay_pipeline)
        overlay_pass.set_bind_group(0, self._highlight_overlay_bind_group)
        overlay_pass.draw(3)
        overlay_pass.end()

    def _set_shadow_bindings(self, encoder, batch, bind_group):
        mesh = batch.mesh
        encoder.set_bind_group(0, bind_group)
        encoder.set_vertex_buffer(0, batch.vertex_buffer)
        if mesh.impostor_kind:
            encoder.set_pipeline(self._impostor_shadow_pipeline(mesh.impostor_kind))
            encoder.set_vertex_buffer(1, batch.impostor_buffer)
            if mesh.impostor_kind == "cylinder":
                encoder.set_vertex_buffer(2, batch.lighting_buffer)
        else:
            encoder.set_pipeline(self._shadow_pipeline)
            encoder.set_vertex_buffer(1, batch.instance_buffer)
        encoder.set_index_buffer(batch.index_buffer, "uint32")

    def _draw_shadow_map(self, command_encoder):
        if self._shadow_params[0] < 0.5 or not self._direct_shadow_dirty:
            return
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": self._shadow_view,
                "depth_clear_value": 1.0,
                "depth_load_op": self.wgpu.LoadOp.clear,
                "depth_store_op": self.wgpu.StoreOp.store,
            },
        )
        for batch in self._gpu_batches:
            mesh = batch.mesh
            if (
                mesh.primitive_topology != "triangle-list"
                or mesh.render_pass != "opaque"
                or not mesh.casts_shadows
            ):
                continue
            self._set_shadow_bindings(render_pass, batch, self._shadow_bind_group)
            render_pass.draw_indexed(mesh.triangles.size, mesh.instance_count)
        render_pass.end()

    def _draw_multishadow_maps(self, command_encoder):
        if self._multishadow_count == 0 or not self._multishadow_dirty:
            return
        for shadow_index in range(self._multishadow_count):
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[],
                depth_stencil_attachment={
                    "view": self._multishadow_layer_views[shadow_index],
                    "depth_clear_value": 1.0,
                    "depth_load_op": self.wgpu.LoadOp.clear,
                    "depth_store_op": self.wgpu.StoreOp.store,
                },
            )
            bind_group = self._multishadow_bind_groups[shadow_index]
            for batch in self._gpu_batches:
                mesh = batch.mesh
                if (
                    mesh.primitive_topology != "triangle-list"
                    or mesh.render_pass != "opaque"
                    or not mesh.casts_shadows
                ):
                    continue
                self._set_shadow_bindings(render_pass, batch, bind_group)
                render_pass.draw_indexed(mesh.triangles.size, mesh.instance_count)
            render_pass.end()

    def _set_mesh_bindings(self, encoder, batch):
        mesh = batch.mesh
        if mesh.impostor_kind:
            encoder.set_pipeline(
                self._impostor_pipeline(mesh.impostor_kind, mesh.render_pass)
            )
            encoder.set_bind_group(0, self.bind_group)
            encoder.set_vertex_buffer(0, batch.vertex_buffer)
            encoder.set_vertex_buffer(1, batch.impostor_buffer)
            encoder.set_vertex_buffer(2, batch.instance_color_buffer)
            if mesh.impostor_kind == "cylinder":
                encoder.set_vertex_buffer(3, batch.secondary_color_buffer)
                encoder.set_vertex_buffer(4, batch.lighting_buffer)
            else:
                encoder.set_vertex_buffer(3, batch.lighting_buffer)
            encoder.set_index_buffer(batch.index_buffer, "uint32")
            return
        encoder.set_pipeline(
            self._pipeline(
                mesh.primitive_topology,
                mesh.render_pass,
                mesh.vertex_colors_are_instance,
            )
        )
        encoder.set_bind_group(0, self.bind_group)
        encoder.set_vertex_buffer(0, batch.vertex_buffer)
        encoder.set_vertex_buffer(1, batch.normal_buffer)
        encoder.set_vertex_buffer(2, batch.instance_buffer)
        encoder.set_vertex_buffer(3, batch.instance_color_buffer)
        encoder.set_vertex_buffer(4, batch.vertex_color_buffer)
        encoder.set_vertex_buffer(5, batch.lighting_buffer)
        encoder.set_index_buffer(batch.index_buffer, "uint32")

    def _bundle(self):
        if not self._gpu_batches:
            return None
        if self._render_bundle is None:
            encoder = self.device.create_render_bundle_encoder(
                color_formats=[self.color_format],
                depth_stencil_format=self.depth_format,
                sample_count=1,
            )
            for batch in self._gpu_batches:
                self._set_mesh_bindings(encoder, batch)
                for _ in range(self.draw_calls):
                    encoder.draw_indexed(
                        batch.mesh.triangles.size, batch.mesh.instance_count
                    )
            self._render_bundle = encoder.finish(label="ChimeraX wgpu mesh draws")
        return self._render_bundle

    def draw_frame(self):
        wgpu = self.wgpu
        width, height = self.canvas.get_physical_size()
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            return
        try:
            target_view = self.context.get_current_texture().create_view()
        except wgpu.DrawCancelled:
            return

        start = perf_counter()
        command_encoder = self.device.create_command_encoder()
        self._draw_shadow_map(command_encoder)
        self._draw_multishadow_maps(command_encoder)
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "resolve_target": None,
                    "clear_value": self.background_color,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self._ensure_depth_view((width, height)),
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        if self._gpu_batches and self.use_render_bundle:
            render_pass.execute_bundles([self._bundle()])
        elif self._gpu_batches:
            for batch in self._gpu_batches:
                self._set_mesh_bindings(render_pass, batch)
                for _ in range(self.draw_calls):
                    render_pass.draw_indexed(
                        batch.mesh.triangles.size, batch.mesh.instance_count
                    )
        render_pass.end()
        self._draw_volumes(command_encoder, target_view, (width, height))
        self._draw_silhouette(command_encoder, target_view)
        self._draw_highlight_overlay(
            command_encoder, target_view, (width, height)
        )
        self.device.queue.submit([command_encoder.finish()])
        self._direct_shadow_dirty = False
        self._multishadow_dirty = False
        if getattr(self.canvas, "manual_present", False):
            self.context.present()
        self._encode_seconds += perf_counter() - start
        self._frame_count += 1

        if self._frame_count == 60:
            elapsed = perf_counter() - self._block_start
            stats = {
                "fps": self._frame_count / elapsed,
                "frame_ms": 1000 * elapsed / self._frame_count,
                "encode_ms": 1000 * self._encode_seconds / self._frame_count,
                "triangles_per_frame": (
                    sum(mesh.rendered_triangle_count for mesh in self.meshes)
                    * self.draw_calls
                ),
                "draw_calls": self.draw_calls,
                "render_bundle": self.use_render_bundle,
                "upload_ms": 1000 * self.upload_seconds,
            }
            if self.stats_callback is not None:
                self.stats_callback(stats)
            self._frame_count = 0
            self._encode_seconds = 0.0
            self._block_start = perf_counter()


def _instance_bounds(vertices, positions):
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    corners = np.asarray(
        [(x, y, z) for x in (lower[0], upper[0])
                   for y in (lower[1], upper[1])
                   for z in (lower[2], upper[2])],
        dtype=np.float32,
    )
    transformed_corners = positions * corners
    return transformed_corners.min(axis=0), transformed_corners.max(axis=0)


def _fit_transform(bounds):
    lower, upper = bounds
    center = 0.5 * (lower + upper)
    radius = float(np.max(upper - lower))
    if radius == 0:
        radius = 1.0
    scale = 1.8 / radius
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = 0.45 * scale
    matrix[0:3, 3] = (-scale * center[0], -scale * center[1],
                      0.5 - 0.45 * scale * center[2])
    return matrix


def _shadow_view_projection(meshes, light_direction):
    """Fit an orthographic WebGPU light frustum around all mesh bounds."""

    if not meshes:
        return np.eye(4, dtype=np.float32)
    lower = np.min(np.asarray([mesh.bounds[0] for mesh in meshes]), axis=0)
    upper = np.max(np.asarray([mesh.bounds[1] for mesh in meshes]), axis=0)
    corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float32,
    )
    direction = np.asarray(light_direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    up_hint = np.asarray((0, 1, 0), dtype=np.float32)
    if abs(float(np.dot(direction, up_hint))) > 0.95:
        up_hint = np.asarray((1, 0, 0), dtype=np.float32)
    right = np.cross(direction, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, direction)
    axes = np.asarray((right, up, direction), dtype=np.float32)
    light_points = corners @ axes.T
    light_lower = light_points.min(axis=0)
    light_upper = light_points.max(axis=0)
    span = np.maximum(light_upper - light_lower, 1e-4)
    padding = 0.05 * span
    light_lower -= padding
    light_upper += padding
    span = light_upper - light_lower

    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, :3] = (2.0 / span[0]) * right
    matrix[0, 3] = -(light_upper[0] + light_lower[0]) / span[0]
    matrix[1, :3] = (2.0 / span[1]) * up
    matrix[1, 3] = -(light_upper[1] + light_lower[1]) / span[1]
    matrix[2, :3] = direction / span[2]
    matrix[2, 3] = -light_lower[2] / span[2]
    matrix[3, 3] = 1.0
    return matrix


def _multishadow_directions(count):
    """Use ChimeraX's same uniform sphere directions as the OpenGL backend."""

    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    from chimerax.geometry import sphere
    return np.asarray(sphere.sphere_points(count), dtype=np.float32)


def chimerax_view_projection(view, window_size):
    """Return ChimeraX's camera transform in WebGPU clip coordinates.

    ChimeraX stores the matrices it passes to OpenGL in column-major byte
    order.  Transposing recovers their conventional mathematical form.  The
    final matrix also remaps OpenGL's -1..1 clip-depth interval to WebGPU's
    0..1 interval.
    """

    camera = view.camera
    near_far = view.near_far_distances(camera, 0)
    projection = np.asarray(
        camera.projection_matrix(near_far, 0, window_size), dtype=np.float32
    ).T
    camera_inverse = camera.get_position(0).inverse(is_orthonormal=True)
    camera_view = np.asarray(camera_inverse.opengl_matrix(), dtype=np.float32).T
    depth_conversion = np.eye(4, dtype=np.float32)
    depth_conversion[2, 2] = 0.5
    depth_conversion[2, 3] = 0.5
    return depth_conversion @ projection @ camera_view


class NativeWgpuCanvas:
    """Mixin implementing direct native presentation for a Qt QWidget.

    The concrete class must inherit from ``QWidget`` before calling
    :meth:`_initialize_native_wgpu_canvas`.  This tiny surface adapter is the
    part rendercanvas normally supplies for us.
    """

    manual_present = True

    def _initialize_native_wgpu_canvas(self):
        import sys
        from Qt.QtCore import Qt

        if not (sys.platform.startswith("darwin") or sys.platform.startswith("win")):
            raise RuntimeError(
                "The rendercanvas-free Qt surface currently supports macOS and "
                "Windows; Linux still needs platform/display handle plumbing."
            )
        try:
            import wgpu
        except ImportError as error:
            raise RuntimeError(
                "The WGPU graphics area needs the optional 'wgpu' Python package."
            ) from error

        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        present_info = {"window": int(self.winId()), "vsync": True}
        self._wgpu_context = wgpu.gpu.get_canvas_context(present_info)
        self.update_wgpu_physical_size()

    def get_wgpu_context(self):
        return self._wgpu_context

    def get_physical_size(self):
        ratio = float(self.devicePixelRatioF())
        return max(0, round(self.width() * ratio)), max(0, round(self.height() * ratio))

    def update_wgpu_physical_size(self):
        size = self.get_physical_size()
        self._wgpu_context.set_physical_size(*size)
        return size

    def paintEngine(self):
        # Qt must not paint into the native view that owns the CAMetalLayer.
        return None


class WgpuPrototypeWindow:
    """Qt window hosting the experimental canvas and benchmark controls."""

    def __init__(self, session, *, draw_calls=1, use_render_bundle=True,
                 continuous=True):
        from Qt.QtCore import Qt, QTimer
        from Qt.QtWidgets import (
            QCheckBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        class _NativeCanvas(NativeWgpuCanvas, QWidget):
            """Minimal scheduled Qt canvas without rendercanvas."""

            def __init__(canvas, parent, size):
                QWidget.__init__(canvas, parent)
                canvas._draw_callback = None
                canvas._draw_pending = False
                canvas._closed = False
                canvas.resize(*size)
                canvas._initialize_native_wgpu_canvas()

            def request_draw(canvas, callback=None):
                if callback is not None:
                    canvas._draw_callback = callback
                if canvas._closed or canvas._draw_pending:
                    return
                canvas._draw_pending = True
                QTimer.singleShot(0, canvas._draw_requested_frame)

            def _draw_requested_frame(canvas):
                canvas._draw_pending = False
                if not canvas._closed and canvas._draw_callback is not None:
                    canvas._draw_callback()

            def resizeEvent(canvas, event):
                QWidget.resizeEvent(canvas, event)
                canvas.update_wgpu_physical_size()
                canvas.request_draw()

            def showEvent(canvas, event):
                QWidget.showEvent(canvas, event)
                canvas.request_draw()

            def paintEvent(canvas, event):
                event.accept()

            def get_closed(canvas):
                return canvas._closed

            def shutdown(canvas):
                if canvas._closed:
                    return
                canvas._closed = True
                canvas._wgpu_context.unconfigure()

        self.session = session
        self._closing = False
        owner = self

        class _PrototypeMainWindow(QMainWindow):
            def closeEvent(window, event):
                owner._begin_close()
                super().closeEvent(event)

        self.window = window = _PrototypeMainWindow()
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.setWindowTitle("ChimeraX wgpu prototype")
        central = QWidget(window)
        layout = QVBoxLayout(central)
        window.setCentralWidget(central)

        self.canvas = canvas = _NativeCanvas(central, size=(900, 700))
        layout.addWidget(canvas, 1)
        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Draw calls:"))
        self.draw_calls = draw_spin = QSpinBox()
        draw_spin.setRange(1, 10000)
        draw_spin.setValue(max(1, int(draw_calls)))
        controls.addWidget(draw_spin)

        self.bundle = bundle_box = QCheckBox("Render bundle")
        bundle_box.setChecked(use_render_bundle)
        controls.addWidget(bundle_box)
        self.continuous = continuous_box = QCheckBox("Continuous")
        continuous_box.setChecked(continuous)
        controls.addWidget(continuous_box)

        draw_button = QPushButton("Draw once")
        reload_button = QPushButton("Reload displayed mesh")
        controls.addWidget(draw_button)
        controls.addWidget(reload_button)
        controls.addStretch(1)

        self.info = QLabel()
        self.stats = QLabel("Waiting for 60 frames...")
        layout.addWidget(self.info)
        layout.addWidget(self.stats)

        mesh = largest_displayed_mesh(session.main_view.drawing)
        self.renderer = WgpuMeshRenderer(
            canvas,
            mesh,
            draw_calls=draw_calls,
            use_render_bundle=use_render_bundle,
            stats_callback=self._show_stats,
        )
        self._show_mesh_info(mesh)
        canvas.request_draw(self.renderer.draw_frame)

        self.timer = QTimer(window)
        self.timer.setInterval(0)
        self.timer.timeout.connect(canvas.request_draw)
        if continuous:
            self.timer.start()

        draw_spin.valueChanged.connect(self.renderer.set_draw_calls)
        bundle_box.toggled.connect(self.renderer.set_use_render_bundle)
        continuous_box.toggled.connect(self._set_continuous)
        draw_button.clicked.connect(lambda: canvas.request_draw())
        reload_button.clicked.connect(self._reload_mesh)
        window.destroyed.connect(self._closed)
        window.resize(960, 820)
        window.show()

    def _set_continuous(self, continuous):
        if continuous:
            self.timer.start()
        else:
            self.timer.stop()

    def _begin_close(self):
        if self._closing:
            return
        self._closing = True
        timer = getattr(self, "timer", None)
        if timer is not None:
            timer.stop()
        canvas = getattr(self, "canvas", None)
        if canvas is not None and not canvas.get_closed():
            canvas.shutdown()

    def _reload_mesh(self):
        try:
            mesh = largest_displayed_mesh(self.session.main_view.drawing)
            self.renderer.upload_mesh(mesh)
            self._show_mesh_info(mesh)
            self.canvas.request_draw()
        except Exception as error:
            self.session.logger.error("Could not reload wgpu mesh: %s" % error)

    def _show_mesh_info(self, mesh):
        reused = "vertices %s, normals %s, indices %s" % (
            "reused" if mesh.reused_vertices else "converted",
            "reused" if mesh.reused_normals else "converted",
            "reused" if mesh.reused_triangles else "converted",
        )
        self.info.setText(
            "%s — %s vertices, %s base triangles × %s instances; %s; "
            "upload %.2f ms; %s; %s"
            % (
                mesh.source_name,
                f"{len(mesh.vertices):,}",
                f"{mesh.triangle_count:,}",
                f"{mesh.instance_count:,}",
                reused,
                1000 * self.renderer.upload_seconds,
                self.renderer.adapter_summary,
                self.renderer.presentation_mode,
            )
        )

    def _show_stats(self, stats):
        mode = "bundle" if stats["render_bundle"] else "Python encoding"
        self.stats.setText(
            "%.1f present fps, %.2f ms/frame (encode %.2f ms), "
            "%s triangles/frame, "
            "%d draws, %s"
            % (
                stats["fps"],
                stats["frame_ms"],
                stats["encode_ms"],
                f"{stats['triangles_per_frame']:,}",
                stats["draw_calls"],
                mode,
            )
        )

    def _closed(self, *_args):
        try:
            self.timer.stop()
        except RuntimeError:
            pass  # Qt may have already destroyed child objects.
        windows = getattr(self.session, "_wgpu_prototype_windows", None)
        if windows is not None and self in windows:
            windows.remove(self)


def show_wgpu_prototype(session, draw_calls=1, render_bundle=True, continuous=True):
    """Open a wgpu window showing the largest displayed triangle drawing."""

    if not session.ui.is_gui:
        from chimerax.core.errors import UserError
        raise UserError("The wgpu prototype requires the ChimeraX graphical interface")
    try:
        window = WgpuPrototypeWindow(
            session,
            draw_calls=draw_calls,
            use_render_bundle=render_bundle,
            continuous=continuous,
        )
    except (RuntimeError, ValueError) as error:
        from chimerax.core.errors import UserError
        raise UserError(str(error)) from error
    windows = getattr(session, "_wgpu_prototype_windows", None)
    if windows is None:
        session._wgpu_prototype_windows = windows = []
    windows.append(window)
    return window


def register_wgpu_command(logger):
    from chimerax.core.commands import BoolArg, CmdDesc, IntArg, register

    desc = CmdDesc(
        keyword=[
            ("draw_calls", IntArg),
            ("render_bundle", BoolArg),
            ("continuous", BoolArg),
        ],
        synopsis="open the experimental wgpu mesh renderer",
    )
    register("wgpu prototype", desc, show_wgpu_prototype, logger=logger)
