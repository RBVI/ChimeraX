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

// mesh.frag -- Standard lit mesh fragment shader (Vulkan port of mesh.metal).

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
layout(constant_id = 6)  const bool USE_TEXTURE_2D = false;
layout(constant_id = 7)  const bool USE_TEXTURE_3D = false;
layout(constant_id = 8)  const bool USE_TEXTURE_CUBEMAP = false;
layout(constant_id = 9)  const bool USE_LIGHTING_NORMALS = false;
layout(constant_id = 10) const bool USE_INSTANCING = false;
layout(constant_id = 11) const bool USE_SHIFT_AND_SCALE = false;
layout(constant_id = 12) const bool USE_COLORMAP = false;
layout(constant_id = 13) const bool USE_ALL_WHITE = false;
layout(constant_id = 14) const bool USE_TRANSPARENT_ONLY = false;
layout(constant_id = 15) const bool USE_OPAQUE_ONLY = false;
layout(constant_id = 16) const bool USE_TEXTURE_3D_AMBIENT = false;
layout(constant_id = 17) const bool USE_DEPTH_PREPASS = false;
layout(constant_id = 19) const bool USE_OIT = false;

// ---------------------------------------------------------------------------
//  Inputs from vertex shader.
// ---------------------------------------------------------------------------
layout(location = 0) in vec3  frag_v;
layout(location = 1) in vec3  frag_N;
layout(location = 2) in vec4  frag_color;
layout(location = 3) in vec2  frag_tex_coord_2d;
layout(location = 4) in vec3  frag_tex_coord_3d;
layout(location = 5) in vec3  frag_shadow_tex_coord;
layout(location = 6) in vec3  frag_ambient_tex_coord;
layout(location = 7) in float frag_clip_distances[8];

// ---------------------------------------------------------------------------
//  Fragment output.
// ---------------------------------------------------------------------------
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_oit_revealage;

// ---------------------------------------------------------------------------
//  Shadow matrix block (SSBO for variable-length array).
// ---------------------------------------------------------------------------
layout(std430, set = 0, binding = 2) readonly buffer ShadowMatrixBlock {
    mat4 shadow_transforms[];
} shadow_block;

// ---------------------------------------------------------------------------
//  Colormap range (push constant or small UBO).
// ---------------------------------------------------------------------------
layout(std140, set = 0, binding = 5) uniform ColormapRangeBlock {
    vec2 colormap_range;
};

// ---------------------------------------------------------------------------
//  Textures -- set 1.
// ---------------------------------------------------------------------------
layout(set = 1, binding = 0) uniform sampler2D      tex2d;
layout(set = 1, binding = 1) uniform sampler3D      tex3d;
layout(set = 1, binding = 2) uniform samplerCube    texcube;
layout(set = 1, binding = 3) uniform sampler2DShadow shadow_map;
layout(set = 1, binding = 4) uniform sampler2DShadow multishadow_map;
layout(set = 1, binding = 5) uniform sampler3D      ambient_tex3d;
layout(set = 1, binding = 6) uniform sampler1D      colormap_tex;

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

    // ---- Base color -------------------------------------------------------
    vec4 fcolor;
    if (USE_TEXTURE_2D) {
        if (USE_COLORMAP) {
            float tvalue = (texture(tex2d, frag_tex_coord_2d).r
                            - colormap_range.x) * colormap_range.y;
            fcolor = texture(colormap_tex, tvalue) * u.color;
        } else {
            fcolor = texture(tex2d, frag_tex_coord_2d) * u.color;
        }
    } else if (USE_TEXTURE_3D) {
        if (USE_COLORMAP) {
            float tvalue = (texture(tex3d, frag_tex_coord_3d).r
                            - colormap_range.x) * colormap_range.y;
            fcolor = texture(colormap_tex, tvalue) * u.color;
        } else {
            fcolor = texture(tex3d, frag_tex_coord_3d) * u.color;
        }
    } else if (USE_TEXTURE_CUBEMAP) {
        fcolor = texture(texcube, frag_tex_coord_3d) * u.color;
    } else if (USE_ALL_WHITE) {
        fcolor = vec4(1.0);
    } else if (USE_VERTEX_COLORS) {
        fcolor = frag_color;
    } else {
        fcolor = u.color;
    }

    // Transparency / opacity pass filtering.
    if (USE_TRANSPARENT_ONLY && fcolor.a >= 0.99)
        discard;
    if (USE_OPAQUE_ONLY && fcolor.a < 0.99)
        discard;

    // ---- Lighting ---------------------------------------------------------
    vec3 Ifrag;
    if (USE_LIGHTING) {
        vec3 N1;
        vec3 v1;
        float key_factor, fill_factor, specular_factor;

        if (USE_LIGHTING_NORMALS) {
            N1 = normalize(frag_N) * (gl_FrontFacing ? 1.0 : -1.0);
            v1 = normalize(frag_v);
            key_factor = max(-dot(N1, lighting.key_light_direction.xyz), 0.0);
            fill_factor = max(-dot(N1, lighting.fill_light_direction.xyz), 0.0);
            vec3 R = normalize(reflect(lighting.key_light_direction.xyz, N1));
            specular_factor = pow(max(-dot(R, v1), 0.0),
                                  lighting.specular_exponent_pad.x);
        } else {
            N1 = vec3(0, 0, 1);
            v1 = vec3(0, 0, 1);
            key_factor = 1.0;
            fill_factor = 1.0;
            specular_factor = 0.0;
        }

        // Single shadow.
        float shadow = 1.0;
        if (USE_SHADOW) {
            vec2 shadow_texel = 1.0 / vec2(textureSize(shadow_map, 0));
            shadow = 0.0;
            for (int i = 0; i < 16; ++i) {
                vec2 offset = poisson[i] * shadow_texel * 1.5;
                shadow += texture(shadow_map,
                    vec3(frag_shadow_tex_coord.xy + offset,
                         frag_shadow_tex_coord.z));
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
                vec3 stc = (stf * vec4(frag_v, 1.0)).xyz;
                float diffuse;
                if (USE_LIGHTING_NORMALS) {
                    vec3 light_dir = u.shadow_depth *
                        vec3(stf[0][2], stf[1][2], stf[2][2]);
                    diffuse = max(-dot(N1, light_dir), 0.0);
                } else {
                    diffuse = 0.25;
                }
                mshadow += diffuse * texture(multishadow_map,
                    vec3(stc.xy, stc.z));
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb,
                                 shadow, mshadow, has_mshadow);

        // Ambient 3-D texture modulation.
        if (USE_TEXTURE_3D_AMBIENT) {
            Ifrag *= texture(ambient_tex3d, frag_ambient_tex_coord).r;
        }
    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (USE_DEPTH_CUE) {
        float d = -frag_v.z;
        float dim = (depth_cue_range.y - d)
                  / (depth_cue_range.y - depth_cue_range.x);
        Ifrag = mix(depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction for angled surfaces -----------------------------
    float a = fcolor.a;
    if (USE_LIGHTING && USE_LIGHTING_NORMALS) {
        vec3 v1 = normalize(frag_v);
        vec3 N1 = normalize(frag_N) * (gl_FrontFacing ? 1.0 : -1.0);
        a = correct_alpha(a, v1, N1);
    }

    if (USE_OIT) {
        // WBOIT accumulation: weight function from McGuire & Bavoil 2013.
        float z = gl_FragCoord.z;
        float w = a * max(1e-2, min(3e3,
            10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 4.0))));
        out_color = vec4(Ifrag * a * w, a * w);
        out_oit_revealage = vec4(a);
    } else {
        out_color = USE_DEPTH_PREPASS ? vec4(0.0) : vec4(Ifrag, a);
    }
}
