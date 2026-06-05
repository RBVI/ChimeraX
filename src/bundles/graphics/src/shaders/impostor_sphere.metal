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

// impostor_sphere.metal -- Impostor sphere rendering via billboard quads and
// per-fragment ray-sphere intersection, with per-fragment depth output.

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
constant bool use_sphere_ao        [[function_constant(18)]];
constant bool use_gtao             [[function_constant(19)]];

constant int MAX_SHADOWS = 256;
struct ShadowMatrixBlock {
    float4x4 shadow_transforms[MAX_SHADOWS];
};

// Per-atom neighbor data for sphere-sphere AO.
struct SphereAOOffset {
    uint start;     // index into neighbor_data array
    uint count;     // number of neighbors for this atom
};

struct SphereAONeighbor {
    float4 pos_radius; // xyz = model-space position, w = radius
};

// ---------------------------------------------------------------------------
//  Vertex attributes -- position.xy are quad corners (-1..1),
//  shift_and_scale.xyz is the sphere centre, .w is the radius.
// ---------------------------------------------------------------------------
struct SphereVertexIn {
    float3 position        [[attribute(0)]];
    float4 vcolor          [[attribute(3)]];
    float4 shift_and_scale [[attribute(4)]];
};

struct SphereVertexOut {
    float4 position      [[position]];
    float3 v;                        // interpolated eye-space position
    float4 color;
    float3 impostor_center;          // sphere centre in eye space
    float  impostor_radius;
    float2 quad_uv;                  // [-1,1] quad corner for debug
    uint   instance_id   [[flat]];   // for indexing neighbor AO buffer
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
//  Vertex shader -- expand unit quad into a camera-facing billboard sized to
//  enclose the sphere projection.
// ---------------------------------------------------------------------------
vertex SphereVertexOut sphere_vertex(
    SphereVertexIn in                 [[stage_in]],
    uint           instance_id        [[instance_id]],
    constant Uniforms &u              [[buffer(0)]])
{
    SphereVertexOut out;
    out.instance_id = instance_id;

    float3 center_eye = (u.model_view_matrix * float4(in.shift_and_scale.xyz, 1.0)).xyz;
    float  radius = in.shift_and_scale.w;

    // Expand the quad on the front side of the sphere.  Clamp the proxy just
    // behind the near plane so it remains rasterizable while staying closer
    // than the ray-cast surface depth written by the fragment shader.
    float center_dist = length(center_eye);
    float3 view_dir = center_dist > 0.0 ? center_eye / center_dist : float3(0.0, 0.0, -1.0);
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    float near_guard = max(u.depth_range.x * 1.001, 0.001);
    float proxy_dist = max(center_dist - radius, near_guard);
    float tangent_denom = sqrt(max(center_dist * center_dist - radius * radius,
                                   radius * radius * 0.01));
    float perspective_extent = radius * proxy_dist / tangent_denom;
    float extent = radius * 1.05 * (ortho ? 1.0 : max(perspective_extent / radius, 1.0));
    float3 proxy_center = ortho
        ? center_eye + float3(0.0, 0.0, radius)
        : center_eye + view_dir * (proxy_dist - center_dist);
    // For off-axis spheres the view-direction proxy shift doesn't bring Z
    // close enough -- the sphere front (center.z + radius) can be closer
    // than the billboard, violating depth(greater).  Clamp proxy Z so the
    // billboard is always in front of or at the sphere's nearest Z.
    if (!ortho)
        proxy_center.z = max(proxy_center.z, center_eye.z + radius * 1.01);
    float4 veye = float4(proxy_center + float3(in.position.xy * extent, 0.0), 1.0);

    out.impostor_center = center_eye;
    out.impostor_radius = radius;
    out.v = veye.xyz;
    out.quad_uv = in.position.xy;

    // Clip planes (emulated).
    if (use_clip_planes) {
        float4 vscene = u.model_matrix * float4(in.shift_and_scale.xyz, 1.0);
        out.clip_distance0 = dot(vscene, u.clip_planes[0]);
            out.clip_distance1 = dot(vscene, u.clip_planes[1]);
            out.clip_distance2 = dot(vscene, u.clip_planes[2]);
            out.clip_distance3 = dot(vscene, u.clip_planes[3]);
            out.clip_distance4 = dot(vscene, u.clip_planes[4]);
            out.clip_distance5 = dot(vscene, u.clip_planes[5]);
            out.clip_distance6 = dot(vscene, u.clip_planes[6]);
            out.clip_distance7 = dot(vscene, u.clip_planes[7]);
    }

    if (use_vertex_colors) {
        out.color = in.vcolor;
    }

    out.position = u.projection_matrix * veye;
    return out;
}

// ---------------------------------------------------------------------------
//  Fragment output with per-fragment depth write.
// ---------------------------------------------------------------------------
struct SphereFragOut {
    float4 color [[color(0)]];
    float  depth [[depth(greater)]];
};

// ---------------------------------------------------------------------------
//  Fragment shader -- ray-sphere intersection, lighting, depth write.
// ---------------------------------------------------------------------------
fragment SphereFragOut sphere_fragment(
    SphereVertexOut in                                [[stage_in]],
    constant Uniforms &u                              [[buffer(0)]],
    constant LightingParams &lighting                 [[buffer(1)]],
    constant ShadowMatrixBlock &shadow_block           [[buffer(2)]],
    constant SphereAOOffset *ao_offsets                [[buffer(3)]],
    constant SphereAONeighbor *ao_neighbors            [[buffer(4)]],
    depth2d<float>    shadow_map                      [[texture(3)]],
    depth2d<float>    multishadow_map                 [[texture(4)]],
    texture2d<float>  gtao_map                        [[texture(7)]],
    sampler           shadow_sampler                  [[sampler(3)]],
    sampler           multishadow_sampler             [[sampler(4)]],
    sampler           gtao_sampler                    [[sampler(7)]])
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

    // ---- Ray-sphere intersection ------------------------------------------
    // projection_matrix[3][3] == 0 for perspective, 1 for orthographic.
    bool ortho = (u.projection_matrix[3][3] != 0.0);
    float3 ray_origin = ortho
        ? float3(in.v.xy, in.impostor_center.z + in.impostor_radius * 2.0)
        : float3(0.0);
    float3 ray_dir = ortho ? float3(0.0, 0.0, -1.0) : normalize(in.v);

    float3 oc = ray_origin - in.impostor_center;
    float b_coeff = dot(oc, ray_dir);
    float c_coeff = dot(oc, oc) - in.impostor_radius * in.impostor_radius;
    float discriminant = b_coeff * b_coeff - c_coeff;
    if (discriminant < 0.0)
        discard_fragment();
    float sqrt_disc = sqrt(discriminant);
    float t_near = -b_coeff - sqrt_disc;
    float t_far = -b_coeff + sqrt_disc;
    float t_hit = (t_near >= 0.0) ? t_near : t_far;
    float near_t = ortho ? 0.0 : (u.depth_range.x * 1.001 / max(-ray_dir.z, 0.0001));
    if (t_hit < near_t && t_far >= near_t)
        t_hit = near_t;
    if (t_hit < 0.0)
        discard_fragment();

    float3 hit = ray_origin + t_hit * ray_dir;
    float3 N1 = normalize(hit - in.impostor_center);

    // ---- Depth ------------------------------------------------------------
    float4 clip_pos = u.projection_matrix * float4(hit, 1.0);
    float ndc_z = clip_pos.z / clip_pos.w;
    float frag_depth = ndc_to_depth(ndc_z, u.depth_range);


    // ---- Base color -------------------------------------------------------
    float4 fcolor = use_vertex_colors ? in.color : u.color;

    if (use_transparent_only && fcolor.a >= 0.99)
        discard_fragment();
    if (use_opaque_only && fcolor.a < 0.99)
        discard_fragment();

    // ---- Lighting ---------------------------------------------------------
    float3 Ifrag;
    if (use_lighting) {
        float3 v1 = normalize(hit);

        // Single shadow -- recompute shadow coord from hit point.
        float shadow = 1.0;
        if (use_shadow) {
            float3 stc = (u.shadow_transform * float4(hit, 1.0)).xyz;
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
                float3 stc = (stf * float4(hit, 1.0)).xyz;
                float diffuse = max(-dot(N1, light_dir), 0.0);
                mshadow += diffuse * multishadow_map.sample_compare(
                    multishadow_sampler, stc.xy, stc.z);
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        // Sphere-sphere AO: exact per-pixel occlusion from neighbor atoms.
        if (use_sphere_ao) {
            SphereAOOffset ao_off = ao_offsets[in.instance_id];
            float sphere_ao = 0.0;
            for (uint n = 0; n < ao_off.count; n++) {
                float4 nb = ao_neighbors[ao_off.start + n].pos_radius;
                // Transform neighbor to eye space.
                float3 nb_eye = (u.model_view_matrix * float4(nb.xyz, 1.0)).xyz;
                float3 to_nb = nb_eye - hit;
                float  d = length(to_nb);
                if (d < 0.001) continue;
                float3 dir = to_nb / d;

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
            float vis = saturate(1.0 - sphere_ao * ao_strength);
            if (has_mshadow) {
                mshadow *= vis;
            } else {
                mshadow = vis;
                has_mshadow = true;
            }
        }

        // Screen-space GTAO: sample the pre-computed AO texture.
        if (use_gtao) {
            float2 screen_uv = in.position.xy
                / float2(gtao_map.get_width(), gtao_map.get_height());
            float ao = gtao_map.sample(gtao_sampler, screen_uv).r;
            if (has_mshadow) {
                mshadow *= ao;
            } else {
                mshadow = ao;
                has_mshadow = true;
            }
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb, lighting,
                                     shadow, mshadow, has_mshadow);

    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (use_depth_cue) {
        float d = -hit.z;
        float dim = (u.depth_cue_range.y - d)
                  / (u.depth_cue_range.y - u.depth_cue_range.x);
        Ifrag = mix(u.depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction -------------------------------------------------
    float a = fcolor.a;
    if (use_lighting) {
        float3 v1 = normalize(hit);
        a = correct_alpha(a, v1, N1);
    }

    SphereFragOut out;
    out.color = use_depth_prepass ? float4(0.0) : float4(Ifrag, a);
    out.depth = frag_depth;
    return out;
}
