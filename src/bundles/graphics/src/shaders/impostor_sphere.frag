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

// impostor_sphere.frag -- Impostor sphere fragment shader (Vulkan port).

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
layout(constant_id = 19) const bool USE_OIT = false;

// ---------------------------------------------------------------------------
//  Inputs from vertex shader.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3  frag_v;
layout(location = 1) in vec4  frag_color;
layout(location = 2) in vec3  frag_impostor_center;
layout(location = 3) in float frag_impostor_radius;
layout(location = 4) in vec2  frag_quad_uv;
layout(location = 5) flat in uint frag_instance_id;
layout(location = 6) in float frag_clip_distances[8];

// ---------------------------------------------------------------------------
//  Fragment outputs -- conservative depth for early-z optimization.
// ---------------------------------------------------------------------------
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_oit_revealage;
layout(depth_greater) out float gl_FragDepth;

// ---------------------------------------------------------------------------
//  Shadow matrix block (SSBO).
// ---------------------------------------------------------------------------
layout(std430, set = 0, binding = 2) readonly buffer ShadowMatrixBlock {
    mat4 shadow_transforms[];
} shadow_block;

// ---------------------------------------------------------------------------
//  Sphere AO buffers.
// ---------------------------------------------------------------------------
struct SphereAOOffset {
    uint start;
    uint count;
};
struct SphereAONeighbor {
    vec4 pos_radius;
};
layout(std430, set = 0, binding = 3) readonly buffer AOOffsetBlock {
    SphereAOOffset ao_offsets[];
};
layout(std430, set = 0, binding = 4) readonly buffer AONeighborBlock {
    SphereAONeighbor ao_neighbors[];
};

// ---------------------------------------------------------------------------
//  Textures -- set 1 (shadow maps only for impostor spheres).
// ---------------------------------------------------------------------------
layout(set = 1, binding = 3) uniform sampler2DShadow shadow_map;
layout(set = 1, binding = 4) uniform sampler2DShadow multishadow_map;

// Poisson disk for shadow PCF (16 taps).
const vec2 poisson[16] = vec2[16](
    vec2( 0.0,     0.0),    vec2(-0.4317,  0.4340),
    vec2( 0.5765, -0.1383), vec2(-0.1706, -0.6389),
    vec2( 0.2955,  0.7103), vec2(-0.7455, -0.1562),
    vec2( 0.8012,  0.3680), vec2(-0.3628, -0.9012),
    vec2(-0.9012,  0.3628), vec2( 0.1562, -0.4549),
    vec2( 0.6389,  0.7455), vec2(-0.7103, -0.5765),
    vec2( 0.9340, -0.1706), vec2(-0.1383, -0.2955),
    vec2( 0.4340,  0.1706), vec2(-0.5765,  0.8012)
);

void main() {
    // Clip-plane discard.
    if (USE_CLIP_PLANES) {
        for (int i = 0; i < u.num_clip_planes; ++i) {
            if (frag_clip_distances[i] < 0.0)
                discard;
        }
    }

    // ---- Ray-sphere intersection ------------------------------------------
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    vec3 ray_origin = ortho
        ? vec3(frag_v.xy, frag_impostor_center.z + frag_impostor_radius * 2.0)
        : vec3(0.0);
    vec3 ray_dir = ortho ? vec3(0.0, 0.0, -1.0) : normalize(frag_v);

    vec3 oc = ray_origin - frag_impostor_center;
    float b_coeff = dot(oc, ray_dir);
    float c_coeff = dot(oc, oc) - frag_impostor_radius * frag_impostor_radius;
    float discriminant = b_coeff * b_coeff - c_coeff;
    if (discriminant < 0.0)
        discard;
    float sqrt_disc = sqrt(discriminant);
    float t_near = -b_coeff - sqrt_disc;
    float t_far  = -b_coeff + sqrt_disc;
    float t_hit  = (t_near >= 0.0) ? t_near : t_far;
    float near_t = ortho ? 0.0 : (depth_range.x * 1.001 / max(-ray_dir.z, 0.0001));
    if (t_hit < near_t && t_far >= near_t)
        t_hit = near_t;
    if (t_hit < 0.0)
        discard;

    vec3 hit = ray_origin + t_hit * ray_dir;
    vec3 N1 = normalize(hit - frag_impostor_center);

    // ---- Depth ------------------------------------------------------------
    vec4 clip_pos = u.projection_matrix * vec4(hit, 1.0);
    float ndc_z = clip_pos.z / clip_pos.w;
    float frag_depth = ndc_to_depth(ndc_z, depth_range);

    // ---- Base color -------------------------------------------------------
    vec4 fcolor = USE_VERTEX_COLORS ? frag_color : u.color;

    if (USE_TRANSPARENT_ONLY && fcolor.a >= 0.99)
        discard;
    if (USE_OPAQUE_ONLY && fcolor.a < 0.99)
        discard;

    // ---- Lighting ---------------------------------------------------------
    vec3 Ifrag;
    if (USE_LIGHTING) {
        vec3 v1 = normalize(hit);

        // Single shadow -- recompute shadow coord from hit point.
        float shadow = 1.0;
        if (USE_SHADOW) {
            vec3 stc = (u.shadow_transform * vec4(hit, 1.0)).xyz;
            vec2 shadow_texel = 1.0 / vec2(textureSize(shadow_map, 0));
            shadow = 0.0;
            for (int i = 0; i < 16; ++i) {
                vec2 offset = poisson[i] * shadow_texel * 1.5;
                shadow += texture(shadow_map,
                    vec3(stc.xy + offset, stc.z));
            }
            shadow /= 16.0;
        }

        // Multishadow.
        float mshadow = 1.0;
        bool has_mshadow = USE_MULTISHADOW;
        if (USE_MULTISHADOW && u.shadow_count > 0) {
            mshadow = 0.0;
            for (int i = 0; i < u.shadow_count; ++i) {
                mat4 stf = shadow_block.shadow_transforms[i];
                vec3 light_dir = u.shadow_depth *
                    vec3(stf[0][2], stf[1][2], stf[2][2]);
                vec3 stc = (stf * vec4(hit, 1.0)).xyz;
                float diffuse = max(-dot(N1, light_dir), 0.0);
                mshadow += diffuse * texture(multishadow_map,
                    vec3(stc.xy, stc.z));
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        // Sphere-sphere AO: exact per-pixel occlusion from neighbor atoms.
        if (USE_SPHERE_AO) {
            SphereAOOffset ao_off = ao_offsets[frag_instance_id];
            float sphere_ao = 0.0;
            for (uint n = 0; n < ao_off.count; n++) {
                vec4 nb = ao_neighbors[ao_off.start + n].pos_radius;
                // Transform neighbor to eye space.
                vec3 nb_eye = (u.model_view_matrix * vec4(nb.xyz, 1.0)).xyz;
                vec3 to_nb = nb_eye - hit;
                float d = length(to_nb);
                if (d < 0.001) continue;
                vec3 dir = to_nb / d;

                // Solid angle of neighbor sphere as seen from hit point.
                float sin_theta = min(nb.w / d, 0.999);
                float cos_theta = sqrt(1.0 - sin_theta * sin_theta);
                float solid_frac = 1.0 - cos_theta;

                // Weight by surface normal -- only the hemisphere above
                // the surface contributes to occlusion.
                float weight = max(dot(N1, dir), 0.0);
                sphere_ao += solid_frac * weight;
            }
            // Combine with existing mshadow (multishadow or default).
            float ao_strength = lighting.sphere_ao_params.x;
            float vis = clamp(1.0 - sphere_ao * ao_strength, 0.0, 1.0);
            if (has_mshadow) {
                mshadow *= vis;
            } else {
                mshadow = vis;
                has_mshadow = true;
            }
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb,
                                 shadow, mshadow, has_mshadow);
    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (USE_DEPTH_CUE) {
        float d = -hit.z;
        float dim = (depth_cue_range.y - d)
                  / (depth_cue_range.y - depth_cue_range.x);
        Ifrag = mix(depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction -------------------------------------------------
    float a = fcolor.a;
    if (USE_LIGHTING) {
        vec3 v1 = normalize(hit);
        a = correct_alpha(a, v1, N1);
    }

    if (USE_OIT) {
        float z = gl_FragCoord.z;
        float w = a * max(1e-2, min(3e3,
            10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 4.0))));
        out_color = vec4(Ifrag * a * w, a * w);
        out_oit_revealage = vec4(a);
    } else {
        out_color = USE_DEPTH_PREPASS ? vec4(0.0) : vec4(Ifrag, a);
    }
    gl_FragDepth = frag_depth;
}
