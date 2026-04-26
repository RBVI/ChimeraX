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

#ifndef CHIMERAX_COMMON_METAL
#define CHIMERAX_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
//  Lighting uniform block -- mirrors GLSL layout(std140) uniform lighting_block.
//  The struct must be packed to 16-byte boundaries so that it can be uploaded
//  as a single constant buffer without per-field padding surprises.
// ---------------------------------------------------------------------------
struct LightingParams {
    float4 key_light_direction;        // .xyz = direction
    float4 key_light_diffuse_color;    // .xyz = color
    float4 key_light_specular_color;   // .xyz = color
    float4 specular_exponent_pad;      // .x = exponent
    float4 fill_light_direction;       // .xyz = direction
    float4 fill_light_diffuse_color;   // .xyz = color
    float4 ambient_color;              // .xyz = color
    float4 sphere_ao_params;           // .x = strength multiplier
};

// ---------------------------------------------------------------------------
//  Per-draw-call uniforms that are shared across all non-postprocess shaders.
// ---------------------------------------------------------------------------
struct Uniforms {
    float4x4 model_view_matrix;
    float4x4 projection_matrix;
    float4x4 model_matrix;         // model -> scene
    float4x4 view_matrix;          // scene -> camera
    float4x4 shadow_transform;
    float4x4 ambient_tex3d_transform;
    float4   color;                // single-color mode
    float3   depth_cue_color;      // float3 is 16 bytes in Metal (implicit pad)
    float2   depth_cue_range;      // start / end camera-z
    float2   depth_range;          // near / far clip distances
    float4   clip_planes[8];
    int      num_clip_planes;
    int      impostor_dashes;      // 0 = solid, >0 = number of dashes
    float    shadow_depth;
    int      shadow_count;
};

// ---------------------------------------------------------------------------
//  Standard mesh vertex attributes -- attribute indices mirror the GLSL
//  layout(location = N) declarations.
// ---------------------------------------------------------------------------
struct MeshVertexIn {
    float3 position   [[attribute(0)]];
    float2 tex_coord  [[attribute(1)]];  // also used as float3 for 3-D / cubemap
    float3 normal     [[attribute(2)]];
    float4 vcolor     [[attribute(3)]];
    float4 shift_and_scale [[attribute(4)]];
    // Instancing placement occupies attributes 5-8 (one float4 per column).
    float4 inst_col0  [[attribute(5)]];
    float4 inst_col1  [[attribute(6)]];
    float4 inst_col2  [[attribute(7)]];
    float4 inst_col3  [[attribute(8)]];
};

// ---------------------------------------------------------------------------
//  Interpolated data passed from vertex to fragment shader.
// ---------------------------------------------------------------------------
struct MeshVertexOut {
    float4 position [[position]];
    float3 v;                  // eye-space position
    float3 N;                  // eye-space normal
    float4 color;
    float2 tex_coord_2d;
    float3 tex_coord_3d;       // also used for cubemap
    float3 shadow_tex_coord;
    float3 ambient_tex_coord;
    float  clip_distance0;     // emulated clip distances (Metal forbids arrays in stage I/O)
    float  clip_distance1;
    float  clip_distance2;
    float  clip_distance3;
    float  clip_distance4;
    float  clip_distance5;
    float  clip_distance6;
    float  clip_distance7;
};

// ---------------------------------------------------------------------------
//  Shared lighting computation.
//
//  Given a surface normal, eye-space direction from camera to fragment, and
//  the lighting parameters, this function computes the same Blinn-style
//  lighting that the GLSL uber-shader produces.
//
//  Parameters
//  ----------
//  N1         : unit surface normal (already flipped for back-facing if needed)
//  view_dir   : unit vector from camera to surface point (normalize(v))
//  fcolor     : base surface color (rgb)
//  lighting   : lighting parameters uniform block
//  shadow     : single shadow factor (0 = fully shadowed, 1 = fully lit)
//  mshadow    : multishadow factor  (pre-divided, 0..1)
//  use_mshadow: whether multishadow is active
// ---------------------------------------------------------------------------
inline float3 compute_lighting(float3 N1,
                               float3 view_dir,
                               float3 fcolor,
                               constant LightingParams &lighting,
                               float shadow,
                               float mshadow,
                               bool use_mshadow)
{
    float key_factor = max(-dot(N1, lighting.key_light_direction.xyz), 0.0);
    float fill_factor = max(-dot(N1, lighting.fill_light_direction.xyz), 0.0);
    float3 R = normalize(reflect(lighting.key_light_direction.xyz, N1));
    float specular_factor = pow(max(-dot(R, view_dir), 0.0),
                                lighting.specular_exponent_pad.x);

    float3 Idiff = fcolor * (lighting.key_light_diffuse_color.xyz * key_factor * shadow
                           + lighting.fill_light_diffuse_color.xyz * fill_factor);
    float3 Ispec = lighting.key_light_specular_color.xyz * specular_factor * shadow;

    float3 Iamb;
    if (use_mshadow) {
        Iamb = fcolor * lighting.ambient_color.xyz * mshadow;
    } else {
        Iamb = fcolor * lighting.ambient_color.xyz;
    }

    return Iamb + Idiff + Ispec;
}

// ---------------------------------------------------------------------------
//  Transparency correction for angled surfaces (same formula as the GLSL
//  shader uses when normals are available).
// ---------------------------------------------------------------------------
inline float correct_alpha(float a, float3 view_dir, float3 N1) {
    if (a >= 1.0) return 1.0;
    return 1.0 - pow(max(1.0 - a, 0.0), 1.0 / max(abs(dot(view_dir, N1)), 0.01));
}

// ---------------------------------------------------------------------------
//  Convert an NDC depth value to a window-space depth using the supplied
//  depth range, replacing gl_DepthRange arithmetic.
//
//  gl_FragDepth = diff*0.5*ndc_z + near + diff*0.5
//  where diff = far - near.
// ---------------------------------------------------------------------------
inline float ndc_to_depth(float ndc_z, float2 depth_range) {
    // Projection matrix is already converted to Metal convention (Z in [0,1]).
    return clamp(ndc_z, 0.0, 1.0);
}

#endif // CHIMERAX_COMMON_METAL
