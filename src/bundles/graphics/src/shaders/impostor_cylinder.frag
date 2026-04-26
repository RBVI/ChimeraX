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

// impostor_cylinder.frag -- Impostor cylinder fragment shader (Vulkan port).

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
//  Inputs from vertex shader.
// ---------------------------------------------------------------------------
layout(location = 0)  in vec3  frag_v;
layout(location = 1)  in vec4  frag_color;
layout(location = 2)  in vec3  frag_impostor_cyl_center;
layout(location = 3)  in vec3  frag_impostor_cyl_axis;
layout(location = 4)  in float frag_impostor_cyl_radius;
layout(location = 5)  in float frag_impostor_cyl_half_length;
layout(location = 6)  in vec4  frag_impostor_cyl_color2;
layout(location = 7)  in vec2  frag_quad_uv;
layout(location = 8)  in float frag_clip_distances[8];

// ---------------------------------------------------------------------------
//  Fragment outputs -- conservative depth for early-z optimization.
// ---------------------------------------------------------------------------
layout(location = 0) out vec4 out_color;
layout(depth_greater) out float gl_FragDepth;

// ---------------------------------------------------------------------------
//  Shadow matrix block (SSBO).
// ---------------------------------------------------------------------------
layout(std430, set = 0, binding = 2) readonly buffer ShadowMatrixBlock {
    mat4 shadow_transforms[];
} shadow_block;

// ---------------------------------------------------------------------------
//  Textures -- set 1 (shadow maps only for impostor cylinders).
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

    // ---- Ray-cylinder intersection ----------------------------------------
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    vec3 ray_origin = ortho
        ? vec3(frag_v.xy,
               frag_impostor_cyl_center.z + frag_impostor_cyl_half_length
               + frag_impostor_cyl_radius * 2.0)
        : vec3(0.0);
    vec3 ray_dir = ortho ? vec3(0.0, 0.0, -1.0) : normalize(frag_v);

    vec3 delta = ray_origin - frag_impostor_cyl_center;
    float d_dot_a     = dot(ray_dir, frag_impostor_cyl_axis);
    float delta_dot_a = dot(delta,   frag_impostor_cyl_axis);
    vec3 ray_perp   = ray_dir - d_dot_a * frag_impostor_cyl_axis;
    vec3 delta_perp = delta   - delta_dot_a * frag_impostor_cyl_axis;

    float qa = dot(ray_perp,   ray_perp);
    float qb = dot(delta_perp, ray_perp);
    float qc = dot(delta_perp, delta_perp)
             - frag_impostor_cyl_radius * frag_impostor_cyl_radius;
    float disc = qb * qb - qa * qc;

    vec3 impostor_hit;
    vec3 impostor_normal;
    bool hit_found = false;

    if (disc >= 0.0) {
        float t = (-qb - sqrt(disc)) / qa;
        vec3 hit_pt = ray_origin + t * ray_dir;
        float along = dot(hit_pt - frag_impostor_cyl_center,
                          frag_impostor_cyl_axis);

        if (abs(along) <= frag_impostor_cyl_half_length) {
            impostor_hit = hit_pt;
            impostor_normal = normalize(
                hit_pt - frag_impostor_cyl_center
                - along * frag_impostor_cyl_axis);
            hit_found = true;
        }
    }

    if (!hit_found) {
        // End-cap intersection (flat discs).
        float cap_t0 = ( frag_impostor_cyl_half_length - delta_dot_a) / d_dot_a;
        float cap_t1 = (-frag_impostor_cyl_half_length - delta_dot_a) / d_dot_a;
        float cap_t = (cap_t0 > 0.0 && cap_t1 > 0.0) ? min(cap_t0, cap_t1)
                     : (cap_t0 > 0.0 ? cap_t0 : cap_t1);
        if (cap_t > 0.0) {
            vec3 cap_hit = ray_origin + cap_t * ray_dir;
            vec3 cap_offset = cap_hit - frag_impostor_cyl_center;
            float cap_along = dot(cap_offset, frag_impostor_cyl_axis);
            vec3 cap_radial = cap_offset - cap_along * frag_impostor_cyl_axis;
            if (dot(cap_radial, cap_radial)
                <= frag_impostor_cyl_radius * frag_impostor_cyl_radius) {
                impostor_hit = cap_hit;
                impostor_normal = sign(cap_along) * frag_impostor_cyl_axis;
                hit_found = true;
            }
        }
    }
    if (!hit_found)
        discard;

    // ---- Dashing ----------------------------------------------------------
    if (u.impostor_dashes > 0) {
        float pos_along = dot(impostor_hit - frag_impostor_cyl_center,
                              frag_impostor_cyl_axis);
        float frac = (pos_along / frag_impostor_cyl_half_length) * 0.5 + 0.5;
        float dash_pos = frac * float(u.impostor_dashes) + 0.75;
        float dash_frac = fract(dash_pos);

        if (dash_frac > 0.5) {
            // In a gap -- try dash-edge caps.
            bool dash_cap_hit = false;
            float edge_offsets[2] = float[2]( 0.5 - dash_frac, 1.0 - dash_frac );
            for (int ei = 0; ei < 2; ++ei) {
                float edge_dash  = dash_pos + edge_offsets[ei];
                float edge_frac  = (edge_dash - 0.75) / float(u.impostor_dashes);
                float edge_along = (edge_frac * 2.0 - 1.0)
                                 * frag_impostor_cyl_half_length;
                float cap_denom = dot(ray_dir, frag_impostor_cyl_axis);
                if (abs(cap_denom) < 0.0001) continue;
                float cap_t = (edge_along
                    - dot(ray_origin - frag_impostor_cyl_center,
                          frag_impostor_cyl_axis)) / cap_denom;
                if (cap_t < 0.0) continue;
                vec3 cap_pt = ray_origin + cap_t * ray_dir;
                vec3 cap_radial = cap_pt - frag_impostor_cyl_center
                                - edge_along * frag_impostor_cyl_axis;
                if (dot(cap_radial, cap_radial)
                    <= frag_impostor_cyl_radius * frag_impostor_cyl_radius) {
                    impostor_hit = cap_pt;
                    impostor_normal = sign(edge_offsets[ei])
                                   * frag_impostor_cyl_axis;
                    dash_cap_hit = true;
                    break;
                }
            }
            if (!dash_cap_hit)
                discard;
        }
    }

    // ---- Depth ------------------------------------------------------------
    if (!ortho) {
        float near_t = depth_range.x * 1.001 / max(-ray_dir.z, 0.0001);
        float hit_t = length(impostor_hit);
        if (hit_t < near_t)
            impostor_hit = ray_dir * near_t;
    }
    vec4 clip_pos = u.projection_matrix * vec4(impostor_hit, 1.0);
    float ndc_z = clip_pos.z / clip_pos.w;
    float frag_depth = ndc_to_depth(ndc_z, depth_range);

    // ---- Half-bond colour -------------------------------------------------
    vec4 fcolor = USE_VERTEX_COLORS ? frag_color : u.color;
    float color_along = dot(impostor_hit - frag_impostor_cyl_center,
                            frag_impostor_cyl_axis);
    if (color_along > 0.0)
        fcolor = frag_impostor_cyl_color2;

    if (USE_TRANSPARENT_ONLY && fcolor.a >= 0.99)
        discard;
    if (USE_OPAQUE_ONLY && fcolor.a < 0.99)
        discard;

    // ---- Lighting ---------------------------------------------------------
    vec3 N1 = impostor_normal;
    vec3 Ifrag;
    if (USE_LIGHTING) {
        vec3 v1 = normalize(impostor_hit);

        // Single shadow.
        float shadow = 1.0;
        if (USE_SHADOW) {
            vec3 stc = (u.shadow_transform * vec4(impostor_hit, 1.0)).xyz;
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
                vec3 stc = (stf * vec4(impostor_hit, 1.0)).xyz;
                float diffuse = max(-dot(N1, light_dir), 0.0);
                mshadow += diffuse * texture(multishadow_map,
                    vec3(stc.xy, stc.z));
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb,
                                 shadow, mshadow, has_mshadow);
    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (USE_DEPTH_CUE) {
        float d = -impostor_hit.z;
        float dim = (depth_cue_range.y - d)
                  / (depth_cue_range.y - depth_cue_range.x);
        Ifrag = mix(depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction -------------------------------------------------
    float a = fcolor.a;
    if (USE_LIGHTING) {
        vec3 v1 = normalize(impostor_hit);
        a = correct_alpha(a, v1, N1);
    }

    out_color = USE_DEPTH_PREPASS ? vec4(0.0) : vec4(Ifrag, a);
    gl_FragDepth = frag_depth;
}
