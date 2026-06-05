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

// pathtracer.metal -- Full path tracer for impostor sphere rendering.
//
// Primary rays from camera, shadow rays for direct illumination,
// cosine-weighted bounce rays for indirect.  Reuses the sphere
// acceleration structure and intersection function from rtao.metal.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// ---------------------------------------------------------------------------
//  Uniforms.
// ---------------------------------------------------------------------------
struct PTUniforms {
    float4x4 inv_projection;       // NDC -> view space
    float4x4 inv_view;             // view space -> world space
    uint     width;
    uint     height;
    uint     frame;                // frame counter for noise
    uint     accum_count;          // temporal accumulation frame count
    // Pack vectors as float4 to avoid float3 alignment issues.
    float4   key_light_dir;        // .xyz = toward light, .w = intensity
    float4   fill_light_dir;       // .xyz = toward fill,  .w = intensity
    float4   ambient_color;        // .xyz = color,        .w = unused
    float4   background_color;     // .xyz = color,        .w = unused
    int      max_bounces;
};

// ---------------------------------------------------------------------------
//  Hash / random helpers (same as rtao.metal).
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

inline float3 cosine_hemisphere(uint pixel_seed, uint ray_index, float3 normal)
{
    uint seed = pixel_seed * 1973u + ray_index * 6133u + 1u;
    float u1 = hash_f(seed);
    float u2 = hash_f(seed * 2654435761u);

    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));

    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return tangent * x + bitangent * y + normal * z;
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
//  Environment lighting for indirect bounces -- sky/ground gradient
//  provides soft ambient illumination from above.
// ---------------------------------------------------------------------------
inline float3 environment_indirect(float3 dir)
{
    float t = 0.5 * (dir.y + 1.0);  // 0 = down, 1 = up
    float3 sky = float3(0.6, 0.75, 0.95);
    float3 ground = float3(0.2, 0.2, 0.2);
    return mix(ground, sky, t);
}

// ---------------------------------------------------------------------------
//  Triangle intersection function.
//  Primitive data: three float4s per triangle (v0, v1, v2; .w unused).
// ---------------------------------------------------------------------------
struct TrianglePrimData {
    float4 v0;
    float4 v1;
    float4 v2;
};

struct TriangleIntersectionResult {
    bool  accept [[accept_intersection]];
    float distance [[distance]];
};

// Möller–Trumbore ray-triangle intersection.
[[intersection(bounding_box)]]
TriangleIntersectionResult triangle_intersection(
    float3 origin               [[origin]],
    float3 direction            [[direction]],
    float  min_distance         [[min_distance]],
    float  max_distance         [[max_distance]],
    const device TrianglePrimData *prim_data [[primitive_data]])
{
    float3 v0 = prim_data->v0.xyz;
    float3 v1 = prim_data->v1.xyz;
    float3 v2 = prim_data->v2.xyz;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(direction, e2);
    float a = dot(e1, h);

    if (abs(a) < 1e-7)
        return {false, 0.0};

    float f = 1.0 / a;
    float3 s = origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return {false, 0.0};

    float3 q = cross(s, e1);
    float v = f * dot(direction, q);
    if (v < 0.0 || u + v > 1.0)
        return {false, 0.0};

    float t = f * dot(e2, q);
    if (t < min_distance || t > max_distance)
        return {false, 0.0};

    return {true, t};
}

// ---------------------------------------------------------------------------
//  Cylinder intersection function.
//  Primitive data: (p0.x, p0.y, p0.z, radius, p1.x, p1.y, p1.z, dashes)
//  = two float4s per cylinder.
// ---------------------------------------------------------------------------
struct CylinderPrimData {
    float4 p0_r;   // .xyz = endpoint 0, .w = radius
    float4 p1_d;   // .xyz = endpoint 1, .w = dash count (0 = solid)
};

struct CylinderIntersectionResult {
    bool  accept [[accept_intersection]];
    float distance [[distance]];
};

// Capped cylinder intersection (Inigo Quilez method).
// Returns nearest t, or -1 on miss.
inline float capped_cyl_isect(float3 ro, float3 rd, float3 a, float3 b, float r)
{
    float3 ba = b - a;
    float3 oc = ro - a;
    float baba = dot(ba, ba);
    float bard = dot(ba, rd);
    float baoc = dot(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot(oc, rd) - baoc * bard;
    float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;

    float h = k1 * k1 - k2 * k0;
    if (h < 0.0) return -1.0;
    h = sqrt(h);

    float t = (-k1 - h) / k2;
    float y = baoc + t * bard;

    // Body hit (near).
    if (y > 0.0 && y < baba) return t;

    // Caps: test whichever side the ray missed.
    float cap = (y <= 0.0) ? 0.0 : baba;
    t = (cap - baoc) / bard;
    if (abs(k2 * t + k1) < h) return t;

    // Body hit (far, for rays starting inside).
    t = (-k1 + h) / k2;
    y = baoc + t * bard;
    if (y > 0.0 && y < baba) return t;

    return -1.0;
}

[[intersection(bounding_box)]]
CylinderIntersectionResult cylinder_intersection(
    float3 origin               [[origin]],
    float3 direction            [[direction]],
    float  min_distance         [[min_distance]],
    float  max_distance         [[max_distance]],
    const device CylinderPrimData *prim_data [[primitive_data]])
{
    float3 a = prim_data->p0_r.xyz;
    float3 b = prim_data->p1_d.xyz;
    float radius = prim_data->p0_r.w;
    float t = capped_cyl_isect(origin, direction, a, b, radius);
    if (t < min_distance || t > max_distance)
        return {false, 0.0};

    int dashes = int(prim_data->p1_d.w);
    if (dashes <= 0) return {true, t};

    // Dash culling: if the body/endcap hit is in a gap, try the two
    // nearest dash-edge cap planes instead.
    float3 ba = b - a;
    float baba = dot(ba, ba);
    float3 hit = origin + direction * t;
    float frac = dot(hit - a, ba) / baba;
    float dash_pos = frac * float(dashes) + 0.75;
    float dash_frac = fract(dash_pos);

    if (dash_frac <= 0.5) return {true, t};  // In a solid segment

    // In a gap - intersect the two nearest dash-edge planes.
    float cap_denom = dot(direction, ba);
    if (abs(cap_denom) < 0.0001 * baba) return {false, 0.0};
    float inv_baba = 1.0 / baba;
    float r2 = radius * radius;
    float edge_offsets[2] = {0.5 - dash_frac, 1.0 - dash_frac};

    for (int ei = 0; ei < 2; ++ei) {
        float edge_dash = dash_pos + edge_offsets[ei];
        float edge_frac = (edge_dash - 0.75) / float(dashes);
        float edge_along = edge_frac * baba;
        float cap_t = (edge_along - dot(origin - a, ba)) / cap_denom;
        if (cap_t < min_distance || cap_t > max_distance) continue;
        float3 cap_pt = origin + direction * cap_t;
        float3 radial = cap_pt - a - ba * (edge_along * inv_baba);
        if (dot(radial, radial) <= r2)
            return {true, cap_t};
    }
    return {false, 0.0};
}

// ---------------------------------------------------------------------------
//  Compute surface normal for a sphere or cylinder hit.
// ---------------------------------------------------------------------------
inline float3 sphere_normal(float3 hit_point, const device float4 *spheres,
                            uint primitive_id)
{
    float4 s = spheres[primitive_id];
    return normalize(hit_point - s.xyz);
}

inline float3 triangle_normal(const device TrianglePrimData *triangles,
                              uint primitive_id)
{
    float3 v0 = triangles[primitive_id].v0.xyz;
    float3 v1 = triangles[primitive_id].v1.xyz;
    float3 v2 = triangles[primitive_id].v2.xyz;
    return normalize(cross(v1 - v0, v2 - v0));
}

inline float3 cylinder_normal(float3 hit_point,
                              const device CylinderPrimData *cylinders,
                              uint primitive_id)
{
    float3 pa = cylinders[primitive_id].p0_r.xyz;
    float3 pb = cylinders[primitive_id].p1_d.xyz;
    float  radius = cylinders[primitive_id].p0_r.w;
    float3 ba = pb - pa;
    float  baba = dot(ba, ba);

    // Project hit onto axis.
    float t_axis = dot(hit_point - pa, ba) / baba;

    // Endcaps.
    if (t_axis <= 0.001) return normalize(-ba);
    if (t_axis >= 0.999) return normalize(ba);

    // Dash-edge caps: if the hit is interior to the cylinder radius,
    // it must be a flat cap face at a dash boundary.
    int dashes = int(cylinders[primitive_id].p1_d.w);
    if (dashes > 0) {
        float3 axis_pt = pa + ba * t_axis;
        float radial_dist2 = dot(hit_point - axis_pt, hit_point - axis_pt);
        float r2 = radius * radius;
        if (radial_dist2 < r2 * 0.99) {
            float frac = t_axis;
            float dash_pos = frac * float(dashes) + 0.75;
            float dash_frac = fract(dash_pos);
            return normalize((dash_frac < 0.25) ? -ba : ba);
        }
    }

    float3 axis_point = pa + ba * t_axis;
    return normalize(hit_point - axis_point);
}

// ---------------------------------------------------------------------------
//  Path tracing kernel.
// ---------------------------------------------------------------------------
[[kernel]]
void pathtracer_kernel(
    uint2 tid                                       [[thread_position_in_grid]],
    texture2d<float, access::write> color_tex        [[texture(0)]],
    texture2d<float, access::read>  prev_tex         [[texture(1)]],
    texture2d<float, access::write> accum_tex        [[texture(2)]],
    primitive_acceleration_structure accel_struct     [[buffer(0)]],
    constant PTUniforms &u                           [[buffer(1)]],
    const device float4 *spheres                     [[buffer(2)]],
    intersection_function_table<>   func_table       [[buffer(3)]],
    const device uchar4 *sphere_colors               [[buffer(4)]],
    const device CylinderPrimData *cylinders         [[buffer(5)]],
    const device uchar4 *cylinder_colors             [[buffer(6)]],
    const device TrianglePrimData *triangles         [[buffer(7)]],
    const device uchar4 *triangle_colors             [[buffer(8)]])
{
    if (tid.x >= u.width || tid.y >= u.height)
        return;

    uint pixel_seed = (tid.x + tid.y * u.width) ^ (u.frame * 7919u);

    // Subpixel jitter for antialiasing - each accumulated frame samples a
    // different position within the pixel, giving progressive SSAA.
    float2 jitter = float2(hash_f(pixel_seed * 3u + 1u),
                           hash_f(pixel_seed * 5u + 3u));
    float2 uv = (float2(tid) + jitter) / float2(u.width, u.height);
    uv.y = 1.0 - uv.y;  // Metal Y=0 is top
    float2 ndc = uv * 2.0 - 1.0;

    float4 view_near = u.inv_projection * float4(ndc, 0.0, 1.0);
    float4 view_far  = u.inv_projection * float4(ndc, 1.0, 1.0);
    view_near /= view_near.w;
    view_far  /= view_far.w;

    float3 ray_origin_world = (u.inv_view * float4(view_near.xyz, 1.0)).xyz;
    float3 ray_target_world = (u.inv_view * float4(view_far.xyz, 1.0)).xyz;
    float3 ray_dir_world = normalize(ray_target_world - ray_origin_world);

    // Set up intersector.
    intersector<> isect;
    isect.force_opacity(forced_opacity::opaque);

    uint ray_counter = 0;

    // Path trace.
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);

    float3 origin = ray_origin_world;
    float3 dir = ray_dir_world;

    for (int bounce = 0; bounce <= u.max_bounces; bounce++) {
        ray r;
        r.origin = origin;
        r.direction = dir;
        r.min_distance = 0.001;
        r.max_distance = 1e10;

        auto result = isect.intersect(r, accel_struct, func_table);
        if (result.type == intersection_type::none) {
            if (bounce == 0) {
                // Primary ray miss: use flat background color.
                radiance = u.background_color.xyz;
            } else {
                // Indirect bounce miss: use sky/ground gradient for
                // soft environmental illumination.
                radiance += throughput * environment_indirect(dir);
            }
            break;
        }

        float3 hit = origin + dir * result.distance;
        float3 N;
        float3 albedo;

        if (result.geometry_id == 0) {
            // Sphere hit.
            N = sphere_normal(hit, spheres, result.primitive_id);
            uchar4 c = sphere_colors[result.primitive_id];
            albedo = float3(c.r, c.g, c.b) / 255.0;
        } else if (result.geometry_id == 1) {
            // Cylinder hit - half-bond coloring.
            N = cylinder_normal(hit, cylinders, result.primitive_id);
            float3 pa = cylinders[result.primitive_id].p0_r.xyz;
            float3 ba = cylinders[result.primitive_id].p1_d.xyz - pa;
            float along = dot(hit - pa, ba);
            bool second_half = (along > dot(ba, ba) * 0.5);
            uchar4 c = cylinder_colors[result.primitive_id * 2 + (second_half ? 1 : 0)];
            albedo = float3(c.r, c.g, c.b) / 255.0;
        } else {
            // Triangle mesh hit.
            N = triangle_normal(triangles, result.primitive_id);
            // Flip normal to face the incoming ray.
            if (dot(N, dir) > 0.0) N = -N;
            uchar4 c = triangle_colors[result.primitive_id];
            albedo = float3(c.r, c.g, c.b) / 255.0;
        }

        // Direct illumination: shadow ray toward key light.
        // Applied at every bounce for correct global illumination.
        float3 key_dir = u.key_light_dir.xyz;
        float  key_int = u.key_light_dir.w;
        float NdotL = max(dot(N, key_dir), 0.0);
        if (NdotL > 0.0) {
            ray shadow_ray;
            shadow_ray.origin = hit + N * 0.002;
            shadow_ray.direction = key_dir;
            shadow_ray.min_distance = 0.0;
            shadow_ray.max_distance = 1e10;

            intersector<> shadow_isect;
            shadow_isect.accept_any_intersection(true);
            shadow_isect.force_opacity(forced_opacity::opaque);
            auto shadow_result = shadow_isect.intersect(
                shadow_ray, accel_struct, func_table);

            if (shadow_result.type == intersection_type::none) {
                radiance += throughput * albedo * NdotL * key_int;
            }
        }

        // Fill and ambient are omitted -- indirect bounces provide
        // the soft illumination that those terms approximate in
        // rasterized rendering.

        // Russian roulette after first bounce.
        if (bounce > 0) {
            float p = max(max(throughput.r, throughput.g), throughput.b);
            if (hash_f(pixel_seed * 3571u + uint(bounce) * 997u) > p)
                break;
            throughput /= p;
        }

        // Set up next bounce: cosine-weighted hemisphere.
        throughput *= albedo;
        origin = hit + N * 0.002;
        dir = cosine_hemisphere(pixel_seed, ray_counter++, N);
        // Cosine hemisphere sampling already includes the cos(theta)/pi PDF
        // cancellation with the Lambertian BRDF (albedo/pi * cos(theta)),
        // so no extra factor is needed.
    }

    // Temporal accumulation.
    float3 color = radiance;
    if (u.accum_count > 1) {
        float3 prev = prev_tex.read(tid).rgb;
        float weight = 1.0 / float(u.accum_count);
        color = mix(prev, color, weight);
    }

    float4 out_color = float4(color, 1.0);
    color_tex.write(out_color, tid);
    accum_tex.write(out_color, tid);
}

// ---------------------------------------------------------------------------
//  Fullscreen blit: copy path tracer output to the framebuffer.
// ---------------------------------------------------------------------------
struct PTBlitVertexIn {
    float3 position  [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct PTBlitVertexOut {
    float4 position     [[position]];
    float2 tex_coord_2d;
};

vertex PTBlitVertexOut pt_blit_vertex(PTBlitVertexIn in [[stage_in]])
{
    PTBlitVertexOut out;
    out.tex_coord_2d = in.tex_coord;
    out.position = float4(in.position, 1.0);
    return out;
}

fragment float4 pt_blit_fragment(
    PTBlitVertexOut in            [[stage_in]],
    texture2d<float> color_tex    [[texture(0)]],
    sampler samp                  [[sampler(0)]])
{
    return color_tex.sample(samp, in.tex_coord_2d);
}
