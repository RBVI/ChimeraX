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

// mesh.vert -- Standard lit mesh vertex shader (Vulkan port of mesh.metal).

#version 450
#include "common.glsl"

// ---------------------------------------------------------------------------
//  Specialization constants -- indices match Metal function_constant values.
// ---------------------------------------------------------------------------
layout(constant_id = 0)  const bool USE_LIGHTING = false;
layout(constant_id = 1)  const bool USE_VERTEX_COLORS = false;
layout(constant_id = 2)  const bool USE_SHADOW = false;
layout(constant_id = 3)  const bool USE_MULTISHADOW = false;
layout(constant_id = 4)  const bool USE_DEPTH_CUE = false;
layout(constant_id = 5)  const bool USE_CLIP_PLANES = false;
layout(constant_id = 6)  const bool USE_TEXTURE_2D = false;
layout(constant_id = 7)  const bool USE_TEXTURE_3D = false;
layout(constant_id = 8)  const bool USE_TEXTURE_CUBEMAP = false;
layout(constant_id = 9)  const bool USE_LIGHTING_NORMALS = false;
layout(constant_id = 10) const bool USE_INSTANCING = false;
layout(constant_id = 11) const bool USE_SHIFT_AND_SCALE = false;
layout(constant_id = 12) const bool USE_COLORMAP = false;
layout(constant_id = 13) const bool USE_ALL_WHITE = false;
layout(constant_id = 14) const bool USE_TRANSPARENT_ONLY = false;
layout(constant_id = 15) const bool USE_OPAQUE_ONLY = false;
layout(constant_id = 16) const bool USE_TEXTURE_3D_AMBIENT = false;
layout(constant_id = 17) const bool USE_DEPTH_PREPASS = false;

// ---------------------------------------------------------------------------
//  Vertex inputs -- attribute locations mirror Metal attribute indices.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 position;
// Texture coordinate is uploaded as 3 floats; .xy is used for 2D textures
// and the full vec3 for 3D / cubemap textures (matches OpenGL's
// USE_TEXTURE_3D layout).
layout(location = 1) in vec3 tex_coord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 vcolor;
layout(location = 4) in vec4 shift_and_scale;
layout(location = 5) in vec4 inst_col0;
layout(location = 6) in vec4 inst_col1;
layout(location = 7) in vec4 inst_col2;
layout(location = 8) in vec4 inst_col3;

// ---------------------------------------------------------------------------
//  Outputs to fragment shader.
// ---------------------------------------------------------------------------
layout(location = 0) out vec3  frag_v;
layout(location = 1) out vec3  frag_N;
layout(location = 2) out vec4  frag_color;
layout(location = 3) out vec2  frag_tex_coord_2d;
layout(location = 4) out vec3  frag_tex_coord_3d;
layout(location = 5) out vec3  frag_shadow_tex_coord;
layout(location = 6) out vec3  frag_ambient_tex_coord;
layout(location = 7) out float frag_clip_distances[8];

void main() {
    // Build local-space position, applying shift_and_scale or instancing.
    vec4 vi;
    if (USE_SHIFT_AND_SCALE) {
        vi = vec4(shift_and_scale.w * position + shift_and_scale.xyz, 1.0);
    } else if (USE_INSTANCING) {
        mat4 placement = mat4(inst_col0, inst_col1, inst_col2, inst_col3);
        vi = placement * vec4(position, 1.0);
    } else {
        vi = vec4(position, 1.0);
    }

    // Clip planes.
    if (USE_CLIP_PLANES) {
        vec4 vscene = u.model_matrix * vi;
        for (int i = 0; i < 8; ++i) {
            frag_clip_distances[i] = dot(vscene, u.clip_planes[i]);
        }
    }

    // Eye-space position.
    vec4 veye = u.model_view_matrix * vi;
    frag_v = veye.xyz;

    // Normal -- transform to eye space.
    if (USE_LIGHTING && USE_LIGHTING_NORMALS) {
        vec3 ni = normal;
        if (USE_INSTANCING) {
            mat4 placement = mat4(inst_col0, inst_col1, inst_col2, inst_col3);
            ni = (placement * vec4(normal, 0.0)).xyz;
        }
        frag_N = normalize((u.model_view_matrix * vec4(ni, 0.0)).xyz);
    }

    // Shadow texture coordinate.
    if (USE_LIGHTING && USE_SHADOW) {
        vec4 stc4 = u.shadow_transform * veye;
        frag_shadow_tex_coord = stc4.xyz;
    }

    // Ambient 3-D texture coordinate.
    if (USE_LIGHTING && USE_TEXTURE_3D_AMBIENT) {
        frag_ambient_tex_coord = (u.ambient_tex3d_transform * vi).xyz;
    }

    // Vertex color.
    if (USE_VERTEX_COLORS) {
        frag_color = vcolor;
    }

    // 2-D texture coordinate.
    if (USE_TEXTURE_2D) {
        frag_tex_coord_2d = tex_coord.xy;
    }

    // 3-D texture coordinate (also used for cubemap).
    if (USE_TEXTURE_3D || USE_TEXTURE_CUBEMAP) {
        frag_tex_coord_3d = tex_coord;
    }

    gl_Position = u.projection_matrix * veye;
}
