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

// impostor_cylinder.metal -- Impostor cylinder rendering via billboard quads,
// ray-cylinder intersection with end caps and dashing, half-bond colouring
// from a packed colour in the instance matrix bottom row, with per-fragment
// depth output.

#include "common.metal"

// Function constants.
constant bool use_lighting      [[function_constant(0)]];
constant bool use_vertex_colors [[function_constant(1)]];
constant bool use_shadow        [[function_constant(2)]];
constant bool use_multishadow   [[function_constant(3)]];
constant bool use_depth_cue     [[function_constant(4)]];
constant bool use_clip_planes   [[function_constant(5)]];
constant bool use_transparent_only [[function_constant(14)]];
constant bool use_opaque_only      [[function_constant(15)]];
constant bool use_depth_prepass    [[function_constant(17)]];

constant int MAX_SHADOWS = 256;
struct ShadowMatrixBlock {
    float4x4 shadow_transforms[MAX_SHADOWS];
};

// ---------------------------------------------------------------------------
//  Vertex attributes -- the instance placement matrix occupies attributes 5-8.
//  The bottom row of that matrix carries the second atom colour (packed as
//  RGBA in [0][3], [1][3], [2][3], [3][3]).
// ---------------------------------------------------------------------------
struct CylinderVertexIn {
    float3 position   [[attribute(0)]];
    float4 vcolor     [[attribute(3)]];
    // Instance placement matrix (4 column vectors).
    float4 inst_col0  [[attribute(5)]];
    float4 inst_col1  [[attribute(6)]];
    float4 inst_col2  [[attribute(7)]];
    float4 inst_col3  [[attribute(8)]];
};

struct CylinderVertexOut {
    float4 position               [[position]];
    float3 v;
    float4 color;
    float3 impostor_cyl_center;
    float3 impostor_cyl_axis;
    float  impostor_cyl_radius;
    float  impostor_cyl_half_length;
    float4 impostor_cyl_color2;
    float2 quad_uv;                  // [-1,1] quad corner for debug
    float  clip_distance0;
    float  clip_distance1;
    float  clip_distance2;
    float  clip_distance3;
    float  clip_distance4;
    float  clip_distance5;
    float  clip_distance6;
    float  clip_distance7;
};

// ---------------------------------------------------------------------------
//  Vertex shader -- billboard expansion that encloses the cylinder projection.
// ---------------------------------------------------------------------------
vertex CylinderVertexOut cylinder_vertex(
    CylinderVertexIn in               [[stage_in]],
    constant Uniforms &u              [[buffer(0)]])
{
    CylinderVertexOut out;

    // Extract the second-atom colour from the bottom row of the instance
    // placement matrix, then zero it so the matrix is a valid affine
    // transform.
    float4 color2 = float4(in.inst_col0.w, in.inst_col1.w,
                           in.inst_col2.w, in.inst_col3.w);
    float4x4 cyl_mat = float4x4(
        float4(in.inst_col0.xyz, 0.0),
        float4(in.inst_col1.xyz, 0.0),
        float4(in.inst_col2.xyz, 0.0),
        float4(in.inst_col3.xyz, 1.0));

    // Cylinder endpoints in eye space (unit cylinder along z, -0.5..+0.5).
    float3 p0_eye = (u.model_view_matrix * cyl_mat * float4(0, 0, -0.5, 1)).xyz;
    float3 p1_eye = (u.model_view_matrix * cyl_mat * float4(0, 0,  0.5, 1)).xyz;
    float3 cyl_axis_eye = p1_eye - p0_eye;
    float  cyl_len = length(cyl_axis_eye);
    float3 cyl_dir = cyl_axis_eye / cyl_len;
    float  cyl_radius = length((u.model_view_matrix * cyl_mat * float4(1, 0, 0, 0)).xyz);
    float3 cyl_center = (p0_eye + p1_eye) * 0.5;
    float  cyl_half_len = cyl_len * 0.5;

    // Build screen-aligned billboard.
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    float3 cyl_view_dir = ortho ? float3(0.0, 0.0, -1.0) : normalize(cyl_center);
    float3 cyl_cross = cross(cyl_dir, cyl_view_dir);
    float  cyl_cross_len = length(cyl_cross);
    float3 cyl_right = cyl_cross_len > 0.001
        ? cyl_cross / cyl_cross_len
        : normalize(cross(cyl_dir, float3(0, 1, 0)));
    float3 cyl_up = cross(cyl_view_dir, cyl_right);

    float3 axis_screen = cyl_dir - cyl_view_dir * dot(cyl_dir, cyl_view_dir);
    float  axis_screen_len = length(axis_screen);
    float3 axis_screen_dir = axis_screen_len > 0.001
        ? axis_screen / axis_screen_len : cyl_up;
    float3 axis_screen_perp = cross(cyl_view_dir, axis_screen_dir);

    float extent_along = cyl_half_len + cyl_radius;
    float extent_perp  = cyl_radius * 1.1;
    float cyl_center_dist = length(cyl_center);
    float front_extent = abs(dot(cyl_dir, cyl_view_dir)) * cyl_half_len + cyl_radius;
    float near_guard = max(u.depth_range.x * 1.001, 0.001);
    float proxy_dist = max(cyl_center_dist - front_extent, near_guard);
    float proxy_shift = proxy_dist - cyl_center_dist;
    float3 cyl_offset = axis_screen_dir  * in.position.y * extent_along
                       + axis_screen_perp * in.position.x * extent_perp
                       + cyl_view_dir * proxy_shift;

    float4 veye = float4(cyl_center + cyl_offset, 1.0);

    out.impostor_cyl_center      = cyl_center;
    out.impostor_cyl_axis        = cyl_dir;
    out.impostor_cyl_radius      = cyl_radius;
    out.impostor_cyl_half_length = cyl_half_len;
    out.impostor_cyl_color2      = color2;
    out.v = veye.xyz;
    out.quad_uv = in.position.xy;

    if (use_vertex_colors) {
        out.color = in.vcolor;
    }

    // Clip planes (emulated).
    if (use_clip_planes) {
        float4 vscene = u.model_matrix * cyl_mat * float4(0, 0, 0, 1);
        out.clip_distance0 = dot(vscene, u.clip_planes[0]);
            out.clip_distance1 = dot(vscene, u.clip_planes[1]);
            out.clip_distance2 = dot(vscene, u.clip_planes[2]);
            out.clip_distance3 = dot(vscene, u.clip_planes[3]);
            out.clip_distance4 = dot(vscene, u.clip_planes[4]);
            out.clip_distance5 = dot(vscene, u.clip_planes[5]);
            out.clip_distance6 = dot(vscene, u.clip_planes[6]);
            out.clip_distance7 = dot(vscene, u.clip_planes[7]);
    }

    out.position = u.projection_matrix * veye;
    return out;
}

// ---------------------------------------------------------------------------
//  Fragment output.
// ---------------------------------------------------------------------------
struct CylinderFragOut {
    float4 color [[color(0)]];
    float  depth [[depth(greater)]];
};

// ---------------------------------------------------------------------------
//  Fragment shader -- ray-cylinder intersection with end caps, dashing,
//  half-bond colouring, lighting, depth write.
// ---------------------------------------------------------------------------
fragment CylinderFragOut cylinder_fragment(
    CylinderVertexOut in                              [[stage_in]],
    constant Uniforms &u                              [[buffer(0)]],
    constant LightingParams &lighting                 [[buffer(1)]],
    constant ShadowMatrixBlock &shadow_block           [[buffer(2)]],
    depth2d<float>    shadow_map                      [[texture(3)]],
    depth2d<float>    multishadow_map                 [[texture(4)]],
    sampler           shadow_sampler                  [[sampler(3)]],
    sampler           multishadow_sampler             [[sampler(4)]])
{
    // Clip-plane discard.
    if (use_clip_planes) {
        float cd[8] = {in.clip_distance0, in.clip_distance1, in.clip_distance2,
                       in.clip_distance3, in.clip_distance4, in.clip_distance5,
                       in.clip_distance6, in.clip_distance7};
            for (int i = 0; i < u.num_clip_planes; ++i) {
                if (cd[i] < 0.0) discard_fragment();
            }
    }

    // ---- Ray-cylinder intersection ----------------------------------------
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    float3 ray_origin = ortho
        ? float3(in.v.xy,
                 in.impostor_cyl_center.z + in.impostor_cyl_half_length
                 + in.impostor_cyl_radius * 2.0)
        : float3(0.0);
    float3 ray_dir = ortho ? float3(0.0, 0.0, -1.0) : normalize(in.v);

    float3 delta = ray_origin - in.impostor_cyl_center;
    float d_dot_a     = dot(ray_dir, in.impostor_cyl_axis);
    float delta_dot_a = dot(delta,   in.impostor_cyl_axis);
    float3 ray_perp   = ray_dir - d_dot_a * in.impostor_cyl_axis;
    float3 delta_perp = delta   - delta_dot_a * in.impostor_cyl_axis;

    float qa = dot(ray_perp,   ray_perp);
    float qb = dot(delta_perp, ray_perp);
    float qc = dot(delta_perp, delta_perp)
             - in.impostor_cyl_radius * in.impostor_cyl_radius;
    float disc = qb * qb - qa * qc;

    float3 impostor_hit;
    float3 impostor_normal;
    bool hit_found = false;

    if (disc >= 0.0) {
        float t = (-qb - sqrt(disc)) / qa;
        float3 hit_pt = ray_origin + t * ray_dir;
        float along = dot(hit_pt - in.impostor_cyl_center, in.impostor_cyl_axis);

        if (abs(along) <= in.impostor_cyl_half_length) {
            impostor_hit = hit_pt;
            impostor_normal = normalize(
                hit_pt - in.impostor_cyl_center - along * in.impostor_cyl_axis);
            hit_found = true;
        }
    }

    if (!hit_found) {
        // End-cap intersection (flat discs).
        float cap_t0 = ( in.impostor_cyl_half_length - delta_dot_a) / d_dot_a;
        float cap_t1 = (-in.impostor_cyl_half_length - delta_dot_a) / d_dot_a;
        float cap_t = (cap_t0 > 0.0 && cap_t1 > 0.0) ? min(cap_t0, cap_t1)
                     : (cap_t0 > 0.0 ? cap_t0 : cap_t1);
        if (cap_t > 0.0) {
            float3 cap_hit = ray_origin + cap_t * ray_dir;
            float3 cap_offset = cap_hit - in.impostor_cyl_center;
            float  cap_along = dot(cap_offset, in.impostor_cyl_axis);
            float3 cap_radial = cap_offset - cap_along * in.impostor_cyl_axis;
            if (dot(cap_radial, cap_radial)
                <= in.impostor_cyl_radius * in.impostor_cyl_radius) {
                impostor_hit = cap_hit;
                impostor_normal = sign(cap_along) * in.impostor_cyl_axis;
                hit_found = true;
            }
        }
    }
    if (!hit_found)
        discard_fragment();

    // ---- Dashing ----------------------------------------------------------
    if (u.impostor_dashes > 0) {
        float pos_along = dot(impostor_hit - in.impostor_cyl_center,
                              in.impostor_cyl_axis);
        float frac = (pos_along / in.impostor_cyl_half_length) * 0.5 + 0.5;
        float dash_pos = frac * float(u.impostor_dashes) + 0.75;
        float dash_frac = fract(dash_pos);

        if (dash_frac > 0.5) {
            // In a gap -- try dash-edge caps.
            bool dash_cap_hit = false;
            float edge_offsets[2] = { 0.5 - dash_frac, 1.0 - dash_frac };
            for (int ei = 0; ei < 2; ++ei) {
                float edge_dash  = dash_pos + edge_offsets[ei];
                float edge_frac  = (edge_dash - 0.75) / float(u.impostor_dashes);
                float edge_along = (edge_frac * 2.0 - 1.0)
                                 * in.impostor_cyl_half_length;
                float cap_denom = dot(ray_dir, in.impostor_cyl_axis);
                if (abs(cap_denom) < 0.0001) continue;
                float cap_t = (edge_along
                    - dot(ray_origin - in.impostor_cyl_center,
                          in.impostor_cyl_axis)) / cap_denom;
                if (cap_t < 0.0) continue;
                float3 cap_pt = ray_origin + cap_t * ray_dir;
                float3 cap_radial = cap_pt - in.impostor_cyl_center
                                  - edge_along * in.impostor_cyl_axis;
                if (dot(cap_radial, cap_radial)
                    <= in.impostor_cyl_radius * in.impostor_cyl_radius) {
                    impostor_hit = cap_pt;
                    impostor_normal = sign(edge_offsets[ei])
                                   * in.impostor_cyl_axis;
                    dash_cap_hit = true;
                    break;
                }
            }
            if (!dash_cap_hit)
                discard_fragment();
        }
    }

    // ---- Depth ------------------------------------------------------------
    if (!ortho) {
        float near_t = u.depth_range.x * 1.001 / max(-ray_dir.z, 0.0001);
        float hit_t = length(impostor_hit);
        if (hit_t < near_t)
            impostor_hit = ray_dir * near_t;
    }
    float4 clip_pos = u.projection_matrix * float4(impostor_hit, 1.0);
    float ndc_z = clip_pos.z / clip_pos.w;
    float frag_depth = ndc_to_depth(ndc_z, u.depth_range);

    // ---- Half-bond colour -------------------------------------------------
    float4 fcolor = use_vertex_colors ? in.color : u.color;
    float color_along = dot(impostor_hit - in.impostor_cyl_center,
                            in.impostor_cyl_axis);
    if (color_along > 0.0)
        fcolor = in.impostor_cyl_color2;

    if (use_transparent_only && fcolor.a >= 0.99)
        discard_fragment();
    if (use_opaque_only && fcolor.a < 0.99)
        discard_fragment();

    // ---- Lighting ---------------------------------------------------------
    float3 N1 = impostor_normal;
    float3 Ifrag;
    if (use_lighting) {
        float3 v1 = normalize(impostor_hit);

        // Single shadow.
        float shadow = 1.0;
        if (use_shadow) {
            float3 stc = (u.shadow_transform * float4(impostor_hit, 1.0)).xyz;
            constexpr float2 poisson[16] = {
                float2( 0.0,     0.0),    float2(-0.4317,  0.4340),
                float2( 0.5765, -0.1383), float2(-0.1706, -0.6389),
                float2( 0.2955,  0.7103), float2(-0.7455, -0.1562),
                float2( 0.8012,  0.3680), float2(-0.3628, -0.9012),
                float2(-0.9012,  0.3628), float2( 0.1562, -0.4549),
                float2( 0.6389,  0.7455), float2(-0.7103, -0.5765),
                float2( 0.9340, -0.1706), float2(-0.1383, -0.2955),
                float2( 0.4340,  0.1706), float2(-0.5765,  0.8012)
            };
            float2 shadow_texel = 1.0 / float2(shadow_map.get_width(),
                                                 shadow_map.get_height());
            shadow = 0.0;
            for (int i = 0; i < 16; ++i) {
                float2 offset = poisson[i] * shadow_texel * 1.5;
                shadow += shadow_map.sample_compare(
                    shadow_sampler, stc.xy + offset, stc.z);
            }
            shadow /= 16.0;
        }

        // Multishadow.
        float mshadow = 1.0;
        bool has_mshadow = use_multishadow;
        if (use_multishadow && u.shadow_count > 0) {
            mshadow = 0.0;
            for (int i = 0; i < u.shadow_count; ++i) {
                float4x4 stf = shadow_block.shadow_transforms[i];
                float3 light_dir = u.shadow_depth *
                    float3(stf[0][2], stf[1][2], stf[2][2]);
                float3 stc = (stf * float4(impostor_hit, 1.0)).xyz;
                float diffuse = max(-dot(N1, light_dir), 0.0);
                mshadow += diffuse * multishadow_map.sample_compare(
                    multishadow_sampler, stc.xy, stc.z);
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb, lighting,
                                 shadow, mshadow, has_mshadow);
    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (use_depth_cue) {
        float d = -impostor_hit.z;
        float dim = (u.depth_cue_range.y - d)
                  / (u.depth_cue_range.y - u.depth_cue_range.x);
        Ifrag = mix(u.depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction -------------------------------------------------
    float a = fcolor.a;
    if (use_lighting) {
        float3 v1 = normalize(impostor_hit);
        a = correct_alpha(a, v1, N1);
    }

    CylinderFragOut out;
    out.color = use_depth_prepass ? float4(0.0) : float4(Ifrag, a);
    out.depth = frag_depth;
    return out;
}
