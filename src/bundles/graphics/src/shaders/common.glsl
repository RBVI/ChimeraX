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

#ifndef CHIMERAX_COMMON_GLSL
#define CHIMERAX_COMMON_GLSL

// ---------------------------------------------------------------------------
//  Lighting uniform block -- packed to std140 layout with 16-byte alignment.
// ---------------------------------------------------------------------------
layout(std140, set = 0, binding = 1) uniform LightingBlock {
    vec4 key_light_direction;        // .xyz = direction
    vec4 key_light_diffuse_color;    // .xyz = color
    vec4 key_light_specular_color;   // .xyz = color
    vec4 specular_exponent_pad;      // .x = exponent
    vec4 fill_light_direction;       // .xyz = direction
    vec4 fill_light_diffuse_color;   // .xyz = color
    vec4 ambient_color;              // .xyz = color
    vec4 sphere_ao_params;           // .x = strength multiplier
} lighting;

// ---------------------------------------------------------------------------
//  Per-draw-call uniforms shared across all non-postprocess shaders.
// ---------------------------------------------------------------------------
layout(std140, set = 0, binding = 0) uniform UniformBlock {
    mat4 model_view_matrix;
    mat4 projection_matrix;
    mat4 model_matrix;              // model -> scene
    mat4 view_matrix;               // scene -> camera
    mat4 shadow_transform;
    mat4 ambient_tex3d_transform;
    vec4 color;                     // single-color mode
    vec4 depth_cue_color_pad;       // .xyz = depth_cue_color
    vec4 depth_range_pack;          // .xy = depth_cue_range, .zw = near/far clip
    vec4 clip_planes[8];
    int  num_clip_planes;
    int  impostor_dashes;           // 0 = solid, >0 = number of dashes
    float shadow_depth;
    int  shadow_count;
    vec4 depth_texture_params;      // xy = coordinate scale, z = depth scale
    vec4 depth_texture_projection;  // x = near, y = far/(far-near)
    vec4 stereo_360_origin_shift;   // xyz = camera origin, w = eye shift
    vec4 stereo_360_vertical;       // xyz = camera vertical axis
} u;

// Tangential eye displacement for omnidirectional stereo. All geometry
// paths use the same scene-space convention, including analytic impostors.
vec4 stereo_360_position(vec4 scene_position) {
    vec3 radial = scene_position.xyz - u.stereo_360_origin_shift.xyz;
    vec3 tangent = cross(radial, u.stereo_360_vertical.xyz);
    float n = length(tangent);
    vec3 eye_axis = n > 0.0 ? tangent / n : vec3(1.0, 0.0, 0.0);
    return vec4(scene_position.xyz - u.stereo_360_origin_shift.w * eye_axis, 1.0);
}

// Convenience accessors that mirror the Metal struct field names.
#define depth_cue_color  (u.depth_cue_color_pad.xyz)
#define depth_cue_range  (u.depth_range_pack.xy)
#define depth_range      (u.depth_range_pack.zw)

// ---------------------------------------------------------------------------
//  Shared lighting computation.
//
//  Given a surface normal, eye-space direction from camera to fragment, and
//  the lighting parameters, this function computes Blinn-style lighting.
// ---------------------------------------------------------------------------
vec3 compute_lighting(vec3 N1, vec3 view_dir, vec3 fcolor,
                      float shadow, float mshadow, bool use_mshadow) {
    float key_factor = max(-dot(N1, lighting.key_light_direction.xyz), 0.0);
    float fill_factor = max(-dot(N1, lighting.fill_light_direction.xyz), 0.0);
    vec3 R = normalize(reflect(lighting.key_light_direction.xyz, N1));
    float specular_factor = pow(max(-dot(R, view_dir), 0.0),
                                lighting.specular_exponent_pad.x);

    vec3 Idiff = fcolor * (lighting.key_light_diffuse_color.xyz * key_factor * shadow
                         + lighting.fill_light_diffuse_color.xyz * fill_factor);
    vec3 Ispec = lighting.key_light_specular_color.xyz * specular_factor * shadow;

    vec3 Iamb;
    if (use_mshadow) {
        Iamb = fcolor * lighting.ambient_color.xyz * mshadow;
    } else {
        Iamb = fcolor * lighting.ambient_color.xyz;
    }

    return Iamb + Idiff + Ispec;
}

// ---------------------------------------------------------------------------
//  Transparency correction for angled surfaces.
// ---------------------------------------------------------------------------
float correct_alpha(float a, vec3 view_dir, vec3 N1) {
    if (a >= 1.0) return 1.0;
    return 1.0 - pow(max(1.0 - a, 0.0), 1.0 / max(abs(dot(view_dir, N1)), 0.01));
}

// ---------------------------------------------------------------------------
//  Convert an NDC depth value to window-space depth.  In Vulkan the
//  projection matrix already maps z to [0,1], so we just clamp.
// ---------------------------------------------------------------------------
float ndc_to_depth(float ndc_z, vec2 dr) {
    return clamp(ndc_z, 0.0, 1.0);
}

#endif // CHIMERAX_COMMON_GLSL
