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

// rtao.metal -- Ray-Traced Ambient Occlusion for impostor sphere rendering.
//
// Uses Metal ray tracing with a custom sphere intersection function to
// cast short AO rays from each visible surface point.  Produces true 3D
// occlusion -- no faceting, no screen-space artifacts, correct burial
// darkening.
//
// Components:
//   1. sphere_intersection -- custom intersection function for ray-sphere
//   2. rtao_kernel -- compute kernel: cast N rays per pixel, output AO
//   3. rtao_composite -- fullscreen fragment: multiply AO into color

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// ---------------------------------------------------------------------------
//  Uniforms for the RTAO compute kernel.
// ---------------------------------------------------------------------------
struct RTAOUniforms {
    float4x4 inv_projection;   // NDC -> view space
    float4x4 inv_view;         // view space -> world space
    uint     width;
    uint     height;
    float    ao_radius;        // max ray distance (world units)
    float    intensity;        // AO strength multiplier
    int      num_rays;         // rays per pixel
    uint     frame;            // frame counter for temporal noise
    uint     accum_count;      // frames accumulated (1 = no history)
};

// ---------------------------------------------------------------------------
//  Hash for random ray direction generation.
// ---------------------------------------------------------------------------
inline float hash_f(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return float(seed) / float(0xFFFFFFFFu);
}

inline float3 random_hemisphere_dir(uint pixel_seed, uint ray_index, float3 normal)
{
    uint seed = pixel_seed * 1973u + ray_index * 6133u + 1u;
    float u1 = hash_f(seed);
    float u2 = hash_f(seed * 2654435761u);

    // Cosine-weighted hemisphere sampling.
    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));

    // Build tangent frame from normal.
    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return tangent * x + bitangent * y + normal * z;
}

// ---------------------------------------------------------------------------
//  Reconstruct view-space position from depth + pixel coordinate.
// ---------------------------------------------------------------------------
inline float3 reconstruct_position(float2 uv, float depth, float4x4 inv_proj)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 view = inv_proj * clip;
    return view.xyz / view.w;
}

// ---------------------------------------------------------------------------
//  Reconstruct view-space normal from depth cross-derivatives.
//  Uses the min-difference method to handle depth discontinuities.
// ---------------------------------------------------------------------------
inline float3 reconstruct_normal(uint2 tid, texture2d<float, access::read> depth_tex,
                                 float3 pos, float4x4 inv_proj,
                                 uint width, uint height)
{
    float2 texel = float2(1.0 / float(width), 1.0 / float(height));
    float2 uv = (float2(tid) + 0.5) * texel;
    uv.y = 1.0 - uv.y;  // Metal Y=0 is top; clip space Y=+1 is top.

    // Sample depth in 4 directions.
    float d_l = depth_tex.read(uint2(max(int(tid.x) - 1, 0), tid.y)).x;
    float d_r = depth_tex.read(uint2(min(tid.x + 1, width - 1), tid.y)).x;
    // tid.y-1 is screen-up in Metal = +Y in clip (after flip).
    float d_up   = depth_tex.read(uint2(tid.x, max(int(tid.y) - 1, 0))).x;
    float d_down = depth_tex.read(uint2(tid.x, min(tid.y + 1, height - 1))).x;

    float depth = depth_tex.read(tid).x;

    // Reconstruct neighbor positions with consistent flipped UVs.
    // tid.y-1 -> higher flipped uv.y, tid.y+1 -> lower flipped uv.y.
    float3 pos_l    = reconstruct_position(uv - float2(texel.x, 0), d_l, inv_proj);
    float3 pos_r    = reconstruct_position(uv + float2(texel.x, 0), d_r, inv_proj);
    float3 pos_up   = reconstruct_position(uv + float2(0, texel.y), d_up, inv_proj);
    float3 pos_down = reconstruct_position(uv - float2(0, texel.y), d_down, inv_proj);

    // Pick derivative with smaller discontinuity (min-difference).
    float3 dx = (abs(d_r - depth) < abs(d_l - depth))
                    ? (pos_r - pos) : (pos - pos_l);
    float3 dy = (abs(d_up - depth) < abs(d_down - depth))
                    ? (pos_up - pos) : (pos - pos_down);

    // cross(dx, dy) points toward the camera (+Z in view space).
    return normalize(cross(dx, dy));
}

// ---------------------------------------------------------------------------
//  Custom intersection function for ray-sphere testing.
// ---------------------------------------------------------------------------
struct SphereIntersectionResult {
    bool  accept [[accept_intersection]];
    float distance [[distance]];
};

[[intersection(bounding_box)]]
SphereIntersectionResult sphere_intersection(
    float3 origin               [[origin]],
    float3 direction            [[direction]],
    float  min_distance         [[min_distance]],
    float  max_distance         [[max_distance]],
    const device float4 *prim_data [[primitive_data]])
{
    float4 sphere = *prim_data;
    float3 center = sphere.xyz;
    float  radius = sphere.w;

    float3 oc = origin - center;
    float b = dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;

    if (disc < 0.0)
        return {false, 0.0};

    float sqrt_disc = sqrt(disc);
    float t = -b - sqrt_disc;
    if (t < min_distance || t > max_distance) {
        t = -b + sqrt_disc;
        if (t < min_distance || t > max_distance)
            return {false, 0.0};
    }

    return {true, t};
}

// ---------------------------------------------------------------------------
//  RTAO compute kernel.
//
//  For each pixel:
//    1. Read depth, skip background
//    2. Reconstruct world-space position and normal
//    3. Cast N short rays in a cosine-weighted hemisphere above the surface
//    4. Count how many hit geometry within ao_radius
//    5. Write AO factor (1 = fully exposed, 0 = fully occluded)
// ---------------------------------------------------------------------------
[[kernel]]
void rtao_kernel(
    uint2 tid                                       [[thread_position_in_grid]],
    texture2d<float, access::read>  depth_tex       [[texture(0)]],
    texture2d<float, access::write> ao_tex          [[texture(1)]],
    texture2d<float, access::read>  prev_tex        [[texture(2)]],
    texture2d<float, access::write> accum_tex       [[texture(3)]],
    primitive_acceleration_structure accel_struct    [[buffer(0)]],
    constant RTAOUniforms &u                        [[buffer(1)]],
    const device float4 *spheres                    [[buffer(2)]],
    intersection_function_table<>   func_table      [[buffer(3)]])
{
    if (tid.x >= u.width || tid.y >= u.height)
        return;

    float depth = depth_tex.read(tid).x;
    if (depth >= 0.9999) {
        ao_tex.write(float4(1.0), tid);
        accum_tex.write(float4(1.0), tid);
        return;
    }

    float2 uv = (float2(tid) + 0.5) / float2(u.width, u.height);
    uv.y = 1.0 - uv.y;  // Metal framebuffer Y=0 is top; projection expects bottom.

    // Reconstruct view-space position and normal.
    float3 pos_view = reconstruct_position(uv, depth, u.inv_projection);
    float3 N_view = reconstruct_normal(tid, depth_tex, pos_view, u.inv_projection,
                                       u.width, u.height);

    // Transform to world space (AS is in world space).
    float3 pos_world = (u.inv_view * float4(pos_view, 1.0)).xyz;
    float3 N_world = normalize((u.inv_view * float4(N_view, 0.0)).xyz);

    // Cast N cosine-weighted hemisphere rays and count occlusion.
    intersector<> isect;
    isect.accept_any_intersection(true);
    isect.force_opacity(forced_opacity::opaque);

    uint pixel_seed = (tid.x + tid.y * u.width) ^ (u.frame * 7919u);
    float3 origin = pos_world + N_world * 0.01;
    int occluded = 0;

    for (int i = 0; i < u.num_rays; i++) {
        ray r;
        r.origin = origin;
        r.direction = random_hemisphere_dir(pixel_seed, uint(i), N_world);
        r.min_distance = 0.0;
        r.max_distance = u.ao_radius;

        auto result = isect.intersect(r, accel_struct, func_table);
        if (result.type != intersection_type::none)
            occluded++;
    }

    float visibility = 1.0 - (float(occluded) / float(u.num_rays));
    float ao = saturate(1.0 - u.intensity * (1.0 - visibility));

    // Temporal accumulation: running average via ping-pong textures.
    if (u.accum_count > 1) {
        float prev = prev_tex.read(tid).r;
        float weight = 1.0 / float(u.accum_count);
        ao = mix(prev, ao, weight);
    }

    accum_tex.write(float4(ao, ao, ao, 1.0), tid);
    ao_tex.write(float4(ao, ao, ao, 1.0), tid);
}

// ---------------------------------------------------------------------------
//  Depth-aware bilateral blur for denoising.
//  Runs as a fullscreen fragment pass: reads noisy AO + depth, writes
//  blurred AO.  Samples in a 7x7 kernel, weighting by spatial distance
//  and depth similarity so edges stay sharp.
// ---------------------------------------------------------------------------
struct RTAOBlurVertexIn {
    float3 position  [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct RTAOBlurVertexOut {
    float4 position     [[position]];
    float2 tex_coord_2d;
};

vertex RTAOBlurVertexOut rtao_blur_vertex(RTAOBlurVertexIn in [[stage_in]])
{
    RTAOBlurVertexOut out;
    out.tex_coord_2d = in.tex_coord;
    out.position = float4(in.position, 1.0);
    return out;
}

struct RTAOBlurUniforms {
    float2 texel_size;    // 1/width, 1/height
    float  depth_threshold;  // depth difference that blocks blur
    float  _pad;
};

fragment float4 rtao_blur_fragment(
    RTAOBlurVertexOut in          [[stage_in]],
    texture2d<float> ao_tex       [[texture(0)]],
    texture2d<float> depth_tex    [[texture(1)]],
    sampler samp                  [[sampler(0)]],
    constant RTAOBlurUniforms &u  [[buffer(0)]])
{
    float2 uv = in.tex_coord_2d;
    float center_ao = ao_tex.sample(samp, uv).r;
    float center_depth = depth_tex.sample(samp, uv).r;

    // Skip background pixels.
    if (center_depth >= 0.9999)
        return float4(center_ao);

    float total_ao = 0.0;
    float total_weight = 0.0;

    // 7x7 bilateral blur.
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            float2 offset = float2(float(dx), float(dy)) * u.texel_size;
            float2 sample_uv = uv + offset;

            float sample_ao = ao_tex.sample(samp, sample_uv).r;
            float sample_depth = depth_tex.sample(samp, sample_uv).r;

            // Spatial weight: Gaussian falloff.
            float dist2 = float(dx * dx + dy * dy);
            float spatial_w = exp(-dist2 / 8.0);

            // Depth weight: reject samples across depth discontinuities.
            float depth_diff = abs(center_depth - sample_depth);
            float depth_w = exp(-depth_diff * depth_diff /
                                (u.depth_threshold * u.depth_threshold));

            float w = spatial_w * depth_w;
            total_ao += sample_ao * w;
            total_weight += w;
        }
    }

    float blurred = total_ao / max(total_weight, 1e-6);
    return float4(blurred, blurred, blurred, 1.0);
}

// ---------------------------------------------------------------------------
//  Composite pass: multiply AO into the scene color buffer.
//  Reuses the fullscreen quad vertex from gtao.metal.
// ---------------------------------------------------------------------------
struct RTAOVertexIn {
    float3 position  [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct RTAOVertexOut {
    float4 position     [[position]];
    float2 tex_coord_2d;
};

vertex RTAOVertexOut rtao_composite_vertex(RTAOVertexIn in [[stage_in]])
{
    RTAOVertexOut out;
    out.tex_coord_2d = in.tex_coord;
    out.position = float4(in.position, 1.0);
    return out;
}

fragment float4 rtao_composite_fragment(
    RTAOVertexOut in              [[stage_in]],
    texture2d<float> ao_tex       [[texture(0)]],
    sampler samp                  [[sampler(0)]])
{
    float ao = ao_tex.sample(samp, in.tex_coord_2d).r;
    // Output black with alpha = (1 - ao).  Standard source-over blend
    // computes: src * srcAlpha + dst * (1 - srcAlpha)
    //         = 0 * (1-ao) + sceneColor * ao = sceneColor * ao.
    return float4(0.0, 0.0, 0.0, 1.0 - ao);
}
