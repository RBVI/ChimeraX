/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2024 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// impostor_cylinder.vert -- Impostor cylinder vertex shader (Vulkan port).

#version 450
#include "common.glsl"

// ---------------------------------------------------------------------------
//  Specialization constants.
// ---------------------------------------------------------------------------
layout(constant_id = 0)  const bool USE_LIGHTING = false;
layout(constant_id = 1)  const bool USE_VERTEX_COLORS = false;
layout(constant_id = 2)  const bool USE_SHADOW = false;
layout(constant_id = 3)  const bool USE_MULTISHADOW = false;
layout(constant_id = 4)  const bool USE_DEPTH_CUE = false;
layout(constant_id = 5)  const bool USE_CLIP_PLANES = false;
layout(constant_id = 14) const bool USE_TRANSPARENT_ONLY = false;
layout(constant_id = 15) const bool USE_OPAQUE_ONLY = false;
layout(constant_id = 17) const bool USE_DEPTH_PREPASS = false;

// ---------------------------------------------------------------------------
//  Vertex inputs.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 position;
layout(location = 3) in vec4 vcolor;
layout(location = 5) in vec4 inst_col0;
layout(location = 6) in vec4 inst_col1;
layout(location = 7) in vec4 inst_col2;
layout(location = 8) in vec4 inst_col3;

// ---------------------------------------------------------------------------
//  Outputs to fragment shader.
// ---------------------------------------------------------------------------
layout(location = 0)  out vec3  frag_v;
layout(location = 1)  out vec4  frag_color;
layout(location = 2)  out vec3  frag_impostor_cyl_center;
layout(location = 3)  out vec3  frag_impostor_cyl_axis;
layout(location = 4)  out float frag_impostor_cyl_radius;
layout(location = 5)  out float frag_impostor_cyl_half_length;
layout(location = 6)  out vec4  frag_impostor_cyl_color2;
layout(location = 7)  out vec2  frag_quad_uv;
layout(location = 8)  out float frag_clip_distances[8];

void main() {
    // Extract the second-atom colour from the bottom row of the instance
    // placement matrix, then zero it so the matrix is a valid affine
    // transform.
    vec4 color2 = vec4(inst_col0.w, inst_col1.w,
                       inst_col2.w, inst_col3.w);
    mat4 cyl_mat = mat4(
        vec4(inst_col0.xyz, 0.0),
        vec4(inst_col1.xyz, 0.0),
        vec4(inst_col2.xyz, 0.0),
        vec4(inst_col3.xyz, 1.0));

    // Cylinder endpoints in eye space (unit cylinder along z, -0.5..+0.5).
    vec3 p0_eye = (u.model_view_matrix * cyl_mat * vec4(0, 0, -0.5, 1)).xyz;
    vec3 p1_eye = (u.model_view_matrix * cyl_mat * vec4(0, 0,  0.5, 1)).xyz;
    vec3 cyl_axis_eye = p1_eye - p0_eye;
    float cyl_len = length(cyl_axis_eye);
    vec3 cyl_dir = cyl_axis_eye / cyl_len;
    float cyl_radius = length((u.model_view_matrix * cyl_mat * vec4(1, 0, 0, 0)).xyz);
    vec3 cyl_center = (p0_eye + p1_eye) * 0.5;
    float cyl_half_len = cyl_len * 0.5;

    // Build screen-aligned billboard.
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    vec3 cyl_view_dir = ortho ? vec3(0.0, 0.0, -1.0) : normalize(cyl_center);
    vec3 cyl_cross = cross(cyl_dir, cyl_view_dir);
    float cyl_cross_len = length(cyl_cross);
    vec3 cyl_right = cyl_cross_len > 0.001
        ? cyl_cross / cyl_cross_len
        : normalize(cross(cyl_dir, vec3(0, 1, 0)));
    vec3 cyl_up = cross(cyl_view_dir, cyl_right);

    vec3 axis_screen = cyl_dir - cyl_view_dir * dot(cyl_dir, cyl_view_dir);
    float axis_screen_len = length(axis_screen);
    vec3 axis_screen_dir = axis_screen_len > 0.001
        ? axis_screen / axis_screen_len : cyl_up;
    vec3 axis_screen_perp = cross(cyl_view_dir, axis_screen_dir);

    float extent_along = cyl_half_len + cyl_radius;
    float extent_perp  = cyl_radius * 1.1;
    float cyl_center_dist = length(cyl_center);
    float front_extent = abs(dot(cyl_dir, cyl_view_dir)) * cyl_half_len + cyl_radius;
    float near_guard = max(depth_range.x * 1.001, 0.001);
    float proxy_dist = max(cyl_center_dist - front_extent, near_guard);
    float proxy_shift = proxy_dist - cyl_center_dist;
    vec3 cyl_offset = axis_screen_dir  * position.y * extent_along
                    + axis_screen_perp * position.x * extent_perp
                    + cyl_view_dir * proxy_shift;

    vec4 veye = vec4(cyl_center + cyl_offset, 1.0);

    frag_impostor_cyl_center      = cyl_center;
    frag_impostor_cyl_axis        = cyl_dir;
    frag_impostor_cyl_radius      = cyl_radius;
    frag_impostor_cyl_half_length = cyl_half_len;
    frag_impostor_cyl_color2      = color2;
    frag_v = veye.xyz;
    frag_quad_uv = position.xy;

    if (USE_VERTEX_COLORS) {
        frag_color = vcolor;
    }

    // Clip planes.
    if (USE_CLIP_PLANES) {
        vec4 vscene = u.model_matrix * cyl_mat * vec4(0, 0, 0, 1);
        for (int i = 0; i < 8; ++i) {
            frag_clip_distances[i] = dot(vscene, u.clip_planes[i]);
        }
    }

    gl_Position = u.projection_matrix * veye;
}
