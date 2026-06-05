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

// gtao.metal -- Ground Truth Ambient Occlusion (horizon-based).
//
// For each pixel, marches the depth buffer in multiple screen-space
// directions to find the horizon angle on each side, then analytically
// integrates the cosine-weighted visible fraction of the hemisphere.
// Using a view-space radius produces both local contact shadows and
// global burial darkening independent of zoom level.
//
// Reference: Jimenez et al., "Practical Realtime Strategies for
// Accurate Indirect Illumination" (SIGGRAPH 2016).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
//  Shared fullscreen-quad vertex types.
// ---------------------------------------------------------------------------
struct GTAOVertexIn {
    float3 position  [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct GTAOVertexOut {
    float4 position     [[position]];
    float2 tex_coord_2d;
};

vertex GTAOVertexOut gtao_vertex(GTAOVertexIn in [[stage_in]])
{
    GTAOVertexOut out;
    out.tex_coord_2d = in.tex_coord;
    out.position = float4(in.position, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
//  GTAO uniforms.
// ---------------------------------------------------------------------------
struct GTAOUniforms {
    float4x4 inv_projection;  // inverse projection matrix
    float2   texel_size;      // 1/width, 1/height
    float    radius;          // AO radius in view-space units (Angstroms)
    float    intensity;       // AO strength (applied as power curve)
    int      directions;      // number of slice directions (typically 4-8)
    int      steps;           // march steps per direction (typically 8-16)
    float    near;            // camera near plane (unused, kept for layout compat)
    float    far;             // camera far plane (unused, kept for layout compat)
    float    bias_frac;       // tangent bias as fraction of radius
};

// ---------------------------------------------------------------------------
//  Utility: spatial hash for per-pixel noise.
// ---------------------------------------------------------------------------
inline float hash(float2 p)
{
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ---------------------------------------------------------------------------
//  Utility: reconstruct view-space position from depth + UV.
// ---------------------------------------------------------------------------
inline float3 reconstruct_position(float2 uv, float depth,
                                   float4x4 inv_proj)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 view = inv_proj * clip;
    return view.xyz / view.w;
}

// ---------------------------------------------------------------------------
//  Reconstruct view-space normal from depth buffer.
//
//  Uses a 4-tap cross pattern, choosing the derivative pair with the
//  smaller depth difference on each axis.  This avoids crossing depth
//  discontinuities at impostor sphere/cylinder edges while still
//  producing smooth normals on continuous surfaces.
// ---------------------------------------------------------------------------
inline float3 reconstruct_normal(float2 uv, float3 P,
                                 texture2d<float> depth_tex,
                                 sampler samp,
                                 float4x4 inv_proj,
                                 float2 texel)
{
    float3 Pr = reconstruct_position(uv + float2(texel.x, 0),
                    depth_tex.sample(samp, uv + float2(texel.x, 0)).x,
                    inv_proj);
    float3 Pl = reconstruct_position(uv - float2(texel.x, 0),
                    depth_tex.sample(samp, uv - float2(texel.x, 0)).x,
                    inv_proj);
    float3 Pu = reconstruct_position(uv + float2(0, texel.y),
                    depth_tex.sample(samp, uv + float2(0, texel.y)).x,
                    inv_proj);
    float3 Pd = reconstruct_position(uv - float2(0, texel.y),
                    depth_tex.sample(samp, uv - float2(0, texel.y)).x,
                    inv_proj);

    float3 dPdx = (abs(Pr.z - P.z) < abs(Pl.z - P.z)) ? (Pr - P) : (P - Pl);
    float3 dPdy = (abs(Pu.z - P.z) < abs(Pd.z - P.z)) ? (Pu - P) : (P - Pd);

    float3 N = normalize(cross(dPdx, dPdy));
    // Ensure normal faces toward camera (+Z in view space).
    if (N.z < 0.0) N = -N;
    return N;
}

// ---------------------------------------------------------------------------
//  Pass 1: Horizon-based GTAO.
//
//  For each pixel we march the depth buffer in D evenly-spaced
//  screen-space directions (each covering both +/- sides).  At each
//  step we compute the elevation angle of the sample above the local
//  tangent plane and track the maximum ("horizon").  Once both horizon
//  angles are known for a slice, the cosine-weighted visible solid
//  angle has a closed-form integral:
//
//      V(h+, h-, n) = sin(b - n) - sin(a - n)
//
//  where a = max(h+, n - pi/2), b = min(pi - h-, n + pi/2), and n is
//  the projected normal angle in the slice.  Averaging over all slices
//  gives the full-hemisphere AO.
// ---------------------------------------------------------------------------
fragment float4 gtao_fragment(
    GTAOVertexOut in                        [[stage_in]],
    texture2d<float> depth_tex              [[texture(0)]],
    sampler          samp                   [[sampler(0)]],
    constant GTAOUniforms &u                [[buffer(0)]])
{
    float2 uv = in.tex_coord_2d;
    float depth = depth_tex.sample(samp, uv).x;
    if (depth >= 0.9999)
        return float4(1.0);

    float3 P = reconstruct_position(uv, depth, u.inv_projection);
    float3 N = reconstruct_normal(uv, P, depth_tex, samp,
                                  u.inv_projection, u.texel_size);

    // ---- View-space radius → screen-space pixel radius -------------------
    // proj[0][0] = fx = 1 / inv_proj[0][0].
    // R_pixels = R_view * fx / (-P.z) * (0.5 / texel_size.x).
    float proj_scale = 0.5 / (u.inv_projection[0][0] * u.texel_size.x);
    float R_pixels = u.radius * proj_scale / (-P.z);
    R_pixels = clamp(R_pixels, 2.0, 512.0);

    float R_view  = u.radius;
    float R2      = R_view * R_view;
    float bias    = R_view * u.bias_frac;
    float step_px = R_pixels / float(u.steps);

    // ---- Per-pixel noise -------------------------------------------------
    float2 pixel = floor(uv / u.texel_size);
    float noise_angle = hash(pixel) * M_PI_F;
    float noise_step  = hash(pixel * 7.23 + 1.7);

    float total_vis = 0.0;

    for (int d = 0; d < u.directions; d++) {
        // Slice angle: evenly spaced over a half-circle, jittered.
        float phi = M_PI_F * (float(d) + 0.5) / float(u.directions)
                  + noise_angle;
        float2 omega = float2(cos(phi), sin(phi));

        // 3D tangent direction in view space for this slice.  The
        // screen-space direction omega maps to view-space through the
        // inverse projection's per-axis scale, which accounts for both
        // FOV and aspect ratio.
        float3 T = normalize(float3(
            omega.x * u.texel_size.x * u.inv_projection[0][0],
            omega.y * u.texel_size.y * u.inv_projection[1][1],
            0.0));

        // Slice bitangent: the component of the view direction
        // perpendicular to T, pointing toward the camera.
        float3 V = normalize(-P);
        float3 B = V - dot(V, T) * T;
        float  B_len = length(B);
        // At extreme grazing angles B can degenerate; fall back to +Z.
        B = (B_len > 0.001) ? (B / B_len) : float3(0, 0, 1);

        // Projected normal angle in the slice (from T toward B).
        float n_angle = atan2(dot(N, B), dot(N, T));

        // ---- March in both directions ------------------------------------
        float h_pos = 0.0;   // horizon on the +omega side
        float h_neg = 0.0;   // horizon on the -omega side

        for (int s = 1; s <= u.steps; s++) {
            float t_px = (float(s) - 0.5 + noise_step) * step_px;
            float2 uv_step = omega * t_px * u.texel_size;

            // +omega direction.
            {
                float2 suv = clamp(uv + uv_step,
                                   float2(0.001), float2(0.999));
                float sd = depth_tex.sample(samp, suv).x;
                if (sd < 0.9999) {
                    float3 S  = reconstruct_position(suv, sd,
                                                     u.inv_projection);
                    float3 ds = S - P;
                    float len2 = dot(ds, ds);
                    float len  = sqrt(len2);
                    if (len > bias) {
                        float h = atan2(dot(ds, B), dot(ds, T));
                        // Quadratic distance falloff to avoid hard
                        // cut-off at the radius boundary.
                        float falloff = saturate(1.0 - len2 / R2);
                        h = mix(h_pos, h, falloff);
                        h_pos = max(h_pos, h);
                    }
                }
            }

            // -omega direction.
            {
                float2 suv = clamp(uv - uv_step,
                                   float2(0.001), float2(0.999));
                float sd = depth_tex.sample(samp, suv).x;
                if (sd < 0.9999) {
                    float3 S  = reconstruct_position(suv, sd,
                                                     u.inv_projection);
                    float3 ds = S - P;
                    float len2 = dot(ds, ds);
                    float len  = sqrt(len2);
                    if (len > bias) {
                        // Horizon measured from -T (the march tangent
                        // on the negative side).
                        float h = atan2(dot(ds, B), dot(ds, -T));
                        float falloff = saturate(1.0 - len2 / R2);
                        h = mix(h_neg, h, falloff);
                        h_neg = max(h_neg, h);
                    }
                }
            }
        }

        h_pos = clamp(h_pos, 0.0, M_PI_F * 0.5);
        h_neg = clamp(h_neg, 0.0, M_PI_F * 0.5);

        // ---- Cosine-weighted visibility integral -------------------------
        // The visible range in the slice is [h_pos, pi - h_neg].
        // Clamp to the hemisphere above the surface normal.
        float a = max(h_pos, n_angle - M_PI_F * 0.5);
        float b = min(M_PI_F - h_neg, n_angle + M_PI_F * 0.5);
        float vis = (b > a) ? (sin(b - n_angle) - sin(a - n_angle))
                             : 0.0;
        total_vis += vis;
    }

    // Max visibility per slice = 2 (full cosine integral over semicircle).
    float ao = total_vis / (float(u.directions) * 2.0);
    ao = saturate(pow(ao, u.intensity));

    return float4(ao, ao, ao, 1.0);
}

// ---------------------------------------------------------------------------
//  Pass 2: composite AO into the scene color buffer.
//
//  Uses Metal framebuffer fetch ([[color(0)]]) to read the existing color
//  and multiply it by the AO factor.  No special blend mode required.
// ---------------------------------------------------------------------------
fragment float4 gtao_composite_fragment(
    GTAOVertexOut in                        [[stage_in]],
    float4        existing_color            [[color(0)]],
    texture2d<float> ao_tex                 [[texture(0)]],
    sampler          samp                   [[sampler(0)]])
{
    float ao = ao_tex.sample(samp, in.tex_coord_2d).r;
    return float4(existing_color.rgb * ao, existing_color.a);
}
