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

// impostor_sphere.vert -- Impostor sphere vertex shader (Vulkan port).

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
layout(constant_id = 18) const bool USE_SPHERE_AO = false;
layout(constant_id = 22) const bool USE_STEREO_360 = false;

// ---------------------------------------------------------------------------
//  Vertex inputs.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 position;
layout(location = 3) in vec4 vcolor;
layout(location = 4) in vec4 shift_and_scale;

// ---------------------------------------------------------------------------
//  Outputs to fragment shader.
// ---------------------------------------------------------------------------
layout(location = 0) out vec3  frag_v;
layout(location = 1) out vec4  frag_color;
layout(location = 2) out vec3  frag_impostor_center;
layout(location = 3) out float frag_impostor_radius;
layout(location = 4) out vec2  frag_quad_uv;
layout(location = 5) flat out uint frag_instance_id;
layout(location = 6) out float frag_clip_distances[8];

void main() {
    frag_instance_id = uint(gl_InstanceIndex);

    vec3 center_eye = (u.model_view_matrix * vec4(shift_and_scale.xyz, 1.0)).xyz;
    if (USE_STEREO_360)
        center_eye = (u.view_matrix * stereo_360_position(
            u.model_matrix * vec4(shift_and_scale.xyz, 1.0))).xyz;
    float radius = shift_and_scale.w;

    // Expand the quad on the front side of the sphere.  Clamp the proxy just
    // behind the near plane so it remains rasterizable while staying closer
    // than the ray-cast surface depth written by the fragment shader.
    float center_dist = length(center_eye);
    vec3 view_dir = center_dist > 0.0 ? center_eye / center_dist : vec3(0.0, 0.0, -1.0);
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    float near_guard = max(depth_range.x * 1.001, 0.001);
    float proxy_dist = max(center_dist - radius, near_guard);
    float tangent_denom = sqrt(max(center_dist * center_dist - radius * radius,
                                   radius * radius * 0.01));
    float perspective_extent = radius * proxy_dist / tangent_denom;
    float extent = radius * 1.05 * (ortho ? 1.0 : max(perspective_extent / radius, 1.0));
    vec3 proxy_center = ortho
        ? center_eye + vec3(0.0, 0.0, radius)
        : center_eye + view_dir * (proxy_dist - center_dist);
    // For off-axis spheres the view-direction proxy shift does not bring Z
    // close enough -- the sphere front (center.z + radius) can be closer
    // than the billboard, violating depth(greater).  Clamp proxy Z so the
    // billboard is always in front of or at the sphere's nearest Z.
    if (!ortho)
        proxy_center.z = max(proxy_center.z, center_eye.z + radius * 1.01);
    vec4 veye = vec4(proxy_center + vec3(position.xy * extent, 0.0), 1.0);

    frag_impostor_center = center_eye;
    frag_impostor_radius = radius;
    frag_v = veye.xyz;
    frag_quad_uv = position.xy;

    // Clip planes.
    if (USE_CLIP_PLANES) {
        vec4 vscene = u.model_matrix * vec4(shift_and_scale.xyz, 1.0);
        for (int i = 0; i < 8; ++i) {
            frag_clip_distances[i] = dot(vscene, u.clip_planes[i]);
        }
    }

    if (USE_VERTEX_COLORS) {
        frag_color = vcolor;
    }

    gl_Position = u.projection_matrix * veye;
}
