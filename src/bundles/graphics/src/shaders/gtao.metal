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

// sao.metal -- Scalable Ambient Obscurance (McGuire et al. 2012).
//
// Two-pass post-process:
//   1. gtao_fragment: reads depth buffer, outputs AO factor per pixel
//   2. gtao_composite_fragment: multiplies AO into the scene color buffer
//
// No surface normals required -- occlusion is computed from depth
// differences alone, avoiding artifacts at depth discontinuities
// between impostor spheres.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
//  Shared fullscreen-quad vertex types (same as postprocess.metal).
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
    float    radius;          // AO radius in pixels
    float    intensity;       // AO strength multiplier
    int      directions;      // number of march directions (typically 4)
    int      steps;           // steps per direction (typically 4)
    float    near;            // camera near plane
    float    far;             // camera far plane
    float    bias_frac;       // bias as fraction of view-space radius
};

// ---------------------------------------------------------------------------
//  Utility: hash for per-pixel sample rotation.
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
//  Spiral sample pattern.  Each point lies on an Archimedean spiral so
//  that the samples cover the disk evenly at any count <= 16.
// ---------------------------------------------------------------------------
constant float2 spiral_samples[16] = {
    float2( 0.100,  0.000), float2(-0.052,  0.181),
    float2(-0.161, -0.186), float2( 0.323,  0.047),
    float2(-0.219,  0.317), float2(-0.075, -0.424),
    float2( 0.400,  0.217), float2(-0.496,  0.098),
    float2( 0.244, -0.490), float2( 0.162,  0.567),
    float2(-0.554,  0.226), float2( 0.568, -0.304),
    float2(-0.207, -0.636), float2(-0.351,  0.599),
    float2( 0.700,  0.160), float2(-0.614, -0.480),
};

// ---------------------------------------------------------------------------
//  Pass 1: Scalable Ambient Obscurance (SAO).
//
//  For each pixel, sample depth at nearby screen positions.  The
//  occlusion contribution from each sample is the depth difference
//  (how much closer the sample surface is) divided by the squared
//  distance (so nearby occluders count more).  This produces smooth
//  contact shadows at small radii AND global burial darkening at
//  large radii, without requiring surface normals.
//
//  Reference: McGuire, Mara, Luebke -- "Scalable Ambient Obscurance"
//  (HPG 2012).
// ---------------------------------------------------------------------------
fragment float4 gtao_fragment(
    GTAOVertexOut in                        [[stage_in]],
    texture2d<float> depth_tex              [[texture(0)]],
    sampler          samp                   [[sampler(0)]],
    constant GTAOUniforms &u                [[buffer(0)]])
{
    float depth = depth_tex.sample(samp, in.tex_coord_2d).x;
    if (depth >= 0.9999)
        return float4(1.0);

    float3 P = reconstruct_position(in.tex_coord_2d, depth, u.inv_projection);

    // Per-pixel noise rotation to hide the spiral pattern.
    float noise_angle = hash(in.tex_coord_2d / u.texel_size) * M_PI_F * 2.0;
    float cn = cos(noise_angle), sn = sin(noise_angle);

    int num_samples = min(u.directions * u.steps, 16);

    // View-space AO radius: convert pixel radius to world units at
    // this depth so the effect scales consistently with distance.
    float R = u.radius * (-P.z) * u.texel_size.x * 2.0;
    float R2 = R * R;
    // Small epsilon to avoid division by zero in the SAO formula.
    float epsilon = 0.0001;
    // Bias: ignore depth differences smaller than this to prevent
    // flat surfaces from self-occluding.
    float bias = R * u.bias_frac;

    float sum = 0.0;

    for (int i = 0; i < num_samples; i++) {
        // Rotate sample offset by per-pixel noise.
        float2 s = spiral_samples[i];
        float2 rotated = float2(s.x * cn - s.y * sn,
                                s.x * sn + s.y * cn);

        float2 sample_uv = in.tex_coord_2d + rotated * u.radius * u.texel_size;
        sample_uv = clamp(sample_uv, float2(0.001), float2(0.999));

        float sd = depth_tex.sample(samp, sample_uv).x;
        if (sd >= 0.9999) continue;

        float3 Q = reconstruct_position(sample_uv, sd, u.inv_projection);
        float3 v = Q - P;

        float vv = dot(v, v);
        float vz = v.z;   // positive = Q closer to camera = potential occluder

        // SAO formula: occlusion ~= max(0, depth_diff - bias) / (dist^2 + eps)
        // Clamped to the AO radius so distant geometry doesn't contribute.
        sum += max(0.0, vz - bias) / (vv + epsilon) * step(vv, R2 * 4.0);
    }

    // Normalize and apply intensity.
    float ao = saturate(1.0 - (R2 * u.intensity * sum) / float(num_samples));

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
