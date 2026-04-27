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

// raycast.frag -- Volume raycast fragment shader. Reconstructs an
// eye-space ray for each pixel from gl_FragCoord + inverse view/projection
// matrices, intersects it with the volume's bounding box, then marches
// through the 3D texture compositing colormapped intensities front-to-back.
// Mirrors the OpenGL fragmentShader.txt USE_VOLUME_RAYCASTING block.

#version 450
#include "common.glsl"

// Per-draw raycast parameters at set 0 binding 6 (new UBO slot just for
// volume raycasting -- mesh / impostor pipelines don't bind here).
layout(std140, set = 0, binding = 6) uniform RaycastBlock {
    mat4 inverse_view_matrix;       // camera -> scene
    mat4 inverse_projection_matrix; // clip -> view
    vec4 background_color;
    vec3 bbox_min;        float step_size;
    vec3 bbox_max;        float _pad1;
    vec3 full_region_min; float _pad2;
    vec3 full_region_max; float _pad3;
    vec2 window_size;     vec2 _pad4;
} raycast;

// Colormap range is shared with the mesh shader at binding 5.
layout(std140, set = 0, binding = 5) uniform ColormapRangeBlock {
    vec2 colormap_range;            // .x = offset, .y = inverse range
};

layout(set = 1, binding = 1) uniform sampler3D tex3d;
layout(set = 1, binding = 6) uniform sampler1D colormap_tex;

layout(location = 0) out vec4 frag_color;

// Cheap hash for stochastic ray-start jitter to break up slice banding.
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    // Reconstruct an eye-space ray from gl_FragCoord. Works for perspective,
    // orthographic, stereo, and 360 cameras since it goes through the
    // (already camera-aware) inverse projection matrix.
    vec2 ndc = 2.0 * gl_FragCoord.xy / raycast.window_size - 1.0;
    vec4 clip_near = vec4(ndc, 0.0, 1.0);   // Vulkan NDC z is [0, 1]
    vec4 clip_far  = vec4(ndc, 1.0, 1.0);
    vec4 view_near = raycast.inverse_projection_matrix * clip_near;
    vec4 view_far  = raycast.inverse_projection_matrix * clip_far;
    view_near /= view_near.w;
    view_far  /= view_far.w;

    vec3 view_ray_origin = view_near.xyz;
    vec3 view_ray_dir    = view_far.xyz - view_near.xyz;

    vec3 ray_origin    = (raycast.inverse_view_matrix * vec4(view_ray_origin, 1.0)).xyz;
    vec3 ray_direction = normalize((raycast.inverse_view_matrix * vec4(view_ray_dir, 0.0)).xyz);

    // Slab-test ray/box intersection.
    vec3 inv_dir = 1.0 / ray_direction;
    vec3 t_bot = inv_dir * (raycast.bbox_min - ray_origin);
    vec3 t_top = inv_dir * (raycast.bbox_max - ray_origin);
    vec3 t_min = min(t_top, t_bot);
    vec3 t_max = max(t_top, t_bot);
    float t_0 = max(t_min.x, max(t_min.y, t_min.z));
    float t_1 = min(t_max.x, min(t_max.y, t_max.z));
    if (t_0 > t_1) discard;

    // Convert entry / exit world points to texture space [0, 1].
    vec3 region_size = raycast.full_region_max - raycast.full_region_min;
    vec3 ray_start = (ray_origin + ray_direction * t_0 - raycast.full_region_min) / region_size;
    vec3 ray_stop  = (ray_origin + ray_direction * t_1 - raycast.full_region_min) / region_size;

    vec3 ray = ray_stop - ray_start;
    float ray_length = length(ray);
    float step_length = raycast.step_size;
    vec3 step_vector = step_length * ray / ray_length;

    float jitter = hash(gl_FragCoord.xy);
    vec3 sample_position = ray_start + step_vector * jitter;

    float grad_offset = min(step_length * 0.5, 0.002);

    frag_color = vec4(0);
    vec3 view_dir = normalize(ray_direction);

    while (ray_length > 0.0 && frag_color.a < 0.99) {
        vec3 sample_pos = clamp(sample_position, vec3(0.001), vec3(0.999));

        float intensity = (texture(tex3d, sample_pos).r - colormap_range.x) * colormap_range.y;
        vec4 cmap_color = texture(colormap_tex, intensity) * u.color;

        if (cmap_color.a > 0.001) {
            // Central-difference gradient for lighting.
            vec3 grad;
            grad.x = texture(tex3d, sample_pos + vec3(grad_offset, 0, 0)).r
                   - texture(tex3d, sample_pos - vec3(grad_offset, 0, 0)).r;
            grad.y = texture(tex3d, sample_pos + vec3(0, grad_offset, 0)).r
                   - texture(tex3d, sample_pos - vec3(0, grad_offset, 0)).r;
            grad.z = texture(tex3d, sample_pos + vec3(0, 0, grad_offset)).r
                   - texture(tex3d, sample_pos - vec3(0, 0, grad_offset)).r;
            float grad_len = length(grad);
            if (grad_len > 0.0001) {
                grad /= grad_len;
                float key_diffuse  = max(0.0, -dot(grad, lighting.key_light_direction.xyz));
                float fill_diffuse = max(0.0, -dot(grad, lighting.fill_light_direction.xyz));
                vec3 R = reflect(lighting.key_light_direction.xyz, grad);
                float specular = pow(max(-dot(R, view_dir), 0.0),
                                     lighting.specular_exponent_pad.x);
                vec3 light_color = key_diffuse  * lighting.key_light_diffuse_color.xyz
                                 + fill_diffuse * lighting.fill_light_diffuse_color.xyz
                                 + lighting.key_light_specular_color.xyz * specular
                                 + lighting.ambient_color.xyz;
                cmap_color.rgb *= light_color;
            } else {
                cmap_color.rgb *= lighting.ambient_color.xyz;
            }

            // Alpha correction so non-uniform step sizes composite consistently.
            float corrected_alpha = 1.0 - pow(1.0 - cmap_color.a, step_length / 0.001);
            cmap_color.a = corrected_alpha;

            // Front-to-back compositing.
            frag_color.rgb += (1.0 - frag_color.a) * cmap_color.a * cmap_color.rgb;
            frag_color.a   += (1.0 - frag_color.a) * cmap_color.a;
        }

        ray_length -= step_length;
        sample_position += step_vector;

        if (any(lessThan(sample_position, vec3(-0.01))) ||
            any(greaterThan(sample_position, vec3(1.01)))) {
            break;
        }
    }

    // Composite over the scene's background color.
    frag_color.rgb += (1.0 - frag_color.a) * raycast.background_color.rgb;
    frag_color.a   += (1.0 - frag_color.a) * raycast.background_color.a;
}
