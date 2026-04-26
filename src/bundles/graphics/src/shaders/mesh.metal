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

// mesh.metal -- Standard lit mesh rendering (the main #else path in the GLSL
// uber-shader).  Handles lighting with normals, vertex colors, instancing,
// shift_and_scale, shadow, multishadow, depth cue, clip planes, and textures.

#include "common.metal"

// ---------------------------------------------------------------------------
//  Function constants -- pipeline specialization replaces the GLSL #ifdef
//  mechanism.  The indices here must match the values the host code passes
//  when building the MTLRenderPipelineState.
// ---------------------------------------------------------------------------
constant bool use_lighting        [[function_constant(0)]];
constant bool use_vertex_colors   [[function_constant(1)]];
constant bool use_shadow          [[function_constant(2)]];
constant bool use_multishadow     [[function_constant(3)]];
constant bool use_depth_cue       [[function_constant(4)]];
constant bool use_clip_planes     [[function_constant(5)]];
constant bool use_texture_2d      [[function_constant(6)]];
constant bool use_texture_3d      [[function_constant(7)]];
constant bool use_texture_cubemap [[function_constant(8)]];
constant bool use_lighting_normals [[function_constant(9)]];
constant bool use_instancing      [[function_constant(10)]];
constant bool use_shift_and_scale [[function_constant(11)]];
constant bool use_colormap        [[function_constant(12)]];
constant bool use_all_white       [[function_constant(13)]];
constant bool use_transparent_only [[function_constant(14)]];
constant bool use_opaque_only     [[function_constant(15)]];
constant bool use_texture_3d_ambient [[function_constant(16)]];
constant bool use_depth_prepass   [[function_constant(17)]];

// Maximum number of shadow lights for the multishadow buffer.
constant int MAX_SHADOWS = 256;

// Shadow matrix block -- mirrors GLSL layout(std140) uniform shadow_matrix_block.
struct ShadowMatrixBlock {
    float4x4 shadow_transforms[MAX_SHADOWS];
};

// ---------------------------------------------------------------------------
//  Vertex shader
// ---------------------------------------------------------------------------
vertex MeshVertexOut mesh_vertex(
    MeshVertexIn in                          [[stage_in]],
    uint         instance_id                 [[instance_id]],
    constant Uniforms &u                     [[buffer(0)]],
    constant LightingParams &lighting        [[buffer(1)]])
{
    MeshVertexOut out;

    // Build the local-space position, applying shift_and_scale or instancing.
    float4 vi;
    if (use_shift_and_scale) {
        vi = float4(in.shift_and_scale.w * in.position + in.shift_and_scale.xyz, 1.0);
    } else if (use_instancing) {
        float4x4 placement = float4x4(in.inst_col0, in.inst_col1,
                                       in.inst_col2, in.inst_col3);
        vi = placement * float4(in.position, 1.0);
    } else {
        vi = float4(in.position, 1.0);
    }

    // Clip planes -- emulated because Metal has no built-in clip distances.
    if (use_clip_planes) {
        float4 vscene = u.model_matrix * vi;
        out.clip_distance0 = dot(vscene, u.clip_planes[0]);
        out.clip_distance1 = dot(vscene, u.clip_planes[1]);
        out.clip_distance2 = dot(vscene, u.clip_planes[2]);
        out.clip_distance3 = dot(vscene, u.clip_planes[3]);
        out.clip_distance4 = dot(vscene, u.clip_planes[4]);
        out.clip_distance5 = dot(vscene, u.clip_planes[5]);
        out.clip_distance6 = dot(vscene, u.clip_planes[6]);
        out.clip_distance7 = dot(vscene, u.clip_planes[7]);
    }

    // Eye-space position.
    float4 veye = u.model_view_matrix * vi;
    out.v = veye.xyz;

    // Normal -- transform to eye space.
    if (use_lighting && use_lighting_normals) {
        float3 ni = in.normal;
        if (use_instancing) {
            float4x4 placement = float4x4(in.inst_col0, in.inst_col1,
                                           in.inst_col2, in.inst_col3);
            ni = (placement * float4(in.normal, 0.0)).xyz;
        }
        out.N = normalize((u.model_view_matrix * float4(ni, 0.0)).xyz);
    }

    // Shadow texture coordinate.
    if (use_lighting && use_shadow) {
        float4 stc4 = u.shadow_transform * veye;
        out.shadow_tex_coord = stc4.xyz;
    }

    // Ambient 3-D texture coordinate.
    if (use_lighting && use_texture_3d_ambient) {
        out.ambient_tex_coord = (u.ambient_tex3d_transform * vi).xyz;
    }

    // Vertex color.
    if (use_vertex_colors) {
        out.color = in.vcolor;
    }

    // 2-D texture coordinate.
    if (use_texture_2d) {
        out.tex_coord_2d = in.tex_coord;
    }

    // 3-D texture coordinate (also used for cubemap).
    if (use_texture_3d || use_texture_cubemap) {
        // attribute(1) is a float2 but the GLSL layout also declares float3
        // at location 1 for 3-D / cubemap.  We reconstruct the third component
        // from the normal attribute slot when needed.
        out.tex_coord_3d = float3(in.tex_coord, in.normal.x);
    }

    out.position = u.projection_matrix * veye;
    return out;
}

// ---------------------------------------------------------------------------
//  Fragment output -- may include a custom depth value.
// ---------------------------------------------------------------------------
struct MeshFragOut {
    float4 color [[color(0)]];
};

// ---------------------------------------------------------------------------
//  Fragment shader
// ---------------------------------------------------------------------------
fragment MeshFragOut mesh_fragment(
    MeshVertexOut in                                [[stage_in]],
    bool          front_facing                      [[front_facing]],
    constant Uniforms &u                            [[buffer(0)]],
    constant LightingParams &lighting               [[buffer(1)]],
    constant ShadowMatrixBlock &shadow_block         [[buffer(2)]],
    texture2d<float>  tex2d                         [[texture(0)]],
    texture3d<float>  tex3d                         [[texture(1)]],
    texturecube<float> texcube                      [[texture(2)]],
    depth2d<float>    shadow_map                    [[texture(3)]],
    depth2d<float>    multishadow_map               [[texture(4)]],
    texture3d<float>  ambient_tex3d                 [[texture(5)]],
    texture1d<float>  colormap_tex                  [[texture(6)]],
    sampler           samp                          [[sampler(0)]],
    sampler           shadow_sampler                [[sampler(3)]],
    sampler           multishadow_sampler           [[sampler(4)]],
    constant float2  &colormap_range                [[buffer(3)]])
{
    // Clip-plane discard (emulated).
    if (use_clip_planes) {
        float cd[8] = {in.clip_distance0, in.clip_distance1, in.clip_distance2,
                       in.clip_distance3, in.clip_distance4, in.clip_distance5,
                       in.clip_distance6, in.clip_distance7};
        for (int i = 0; i < u.num_clip_planes; ++i) {
            if (cd[i] < 0.0)
                discard_fragment();
        }
    }

    // ---- Base color -------------------------------------------------------
    float4 fcolor;
    if (use_texture_2d) {
        if (use_colormap) {
            float tvalue = (tex2d.sample(samp, in.tex_coord_2d).r
                            - colormap_range.x) * colormap_range.y;
            fcolor = colormap_tex.sample(samp, tvalue) * u.color;
        } else {
            fcolor = tex2d.sample(samp, in.tex_coord_2d) * u.color;
        }
    } else if (use_texture_3d) {
        if (use_colormap) {
            float tvalue = (tex3d.sample(samp, in.tex_coord_3d).r
                            - colormap_range.x) * colormap_range.y;
            fcolor = colormap_tex.sample(samp, tvalue) * u.color;
        } else {
            fcolor = tex3d.sample(samp, in.tex_coord_3d) * u.color;
        }
    } else if (use_texture_cubemap) {
        fcolor = texcube.sample(samp, in.tex_coord_3d) * u.color;
    } else if (use_all_white) {
        fcolor = float4(1.0);
    } else if (use_vertex_colors) {
        fcolor = in.color;
    } else {
        fcolor = u.color;
    }

    // Transparency / opacity pass filtering.
    if (use_transparent_only && fcolor.a >= 0.99)
        discard_fragment();
    if (use_opaque_only && fcolor.a < 0.99)
        discard_fragment();

    // ---- Lighting ---------------------------------------------------------
    float3 Ifrag;
    if (use_lighting) {
        float3 N1;
        float3 v1;
        float key_factor, fill_factor, specular_factor;

        if (use_lighting_normals) {
            N1 = normalize(in.N) * (front_facing ? 1.0 : -1.0);
            v1 = normalize(in.v);
            key_factor = max(-dot(N1, lighting.key_light_direction.xyz), 0.0);
            fill_factor = max(-dot(N1, lighting.fill_light_direction.xyz), 0.0);
            float3 R = normalize(reflect(lighting.key_light_direction.xyz, N1));
            specular_factor = pow(max(-dot(R, v1), 0.0),
                                  lighting.specular_exponent_pad.x);
        } else {
            N1 = float3(0, 0, 1);
            v1 = float3(0, 0, 1);
            key_factor = 1.0;
            fill_factor = 1.0;
            specular_factor = 0.0;
        }

        // Single shadow.
        float shadow = 1.0;
        if (use_shadow) {
            // Poisson-disk PCF (16 taps) -- same pattern as the GLSL shader.
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
                    shadow_sampler,
                    in.shadow_tex_coord.xy + offset,
                    in.shadow_tex_coord.z);
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
                float3 stc = (stf * float4(in.v, 1.0)).xyz;
                float diffuse;
                if (use_lighting_normals) {
                    float3 light_dir = u.shadow_depth *
                        float3(stf[0][2], stf[1][2], stf[2][2]);
                    diffuse = max(-dot(N1, light_dir), 0.0);
                } else {
                    diffuse = 0.25;
                }
                mshadow += diffuse * multishadow_map.sample_compare(
                    multishadow_sampler, stc.xy, stc.z);
            }
            mshadow /= (0.25 * float(u.shadow_count));
        } else {
            has_mshadow = false;
        }

        Ifrag = compute_lighting(N1, v1, fcolor.rgb, lighting,
                                 shadow, mshadow, has_mshadow);

        // Ambient 3-D texture modulation.
        if (use_texture_3d_ambient) {
            Ifrag *= ambient_tex3d.sample(samp, in.ambient_tex_coord).r;
        }
    } else {
        Ifrag = fcolor.rgb;
    }

    // ---- Depth cue --------------------------------------------------------
    if (use_depth_cue) {
        float d = -in.v.z;
        float dim = (u.depth_cue_range.y - d)
                  / (u.depth_cue_range.y - u.depth_cue_range.x);
        Ifrag = mix(u.depth_cue_color, Ifrag, clamp(dim, 0.0, 1.0));
    }

    // ---- Alpha correction for angled surfaces -----------------------------
    float a = fcolor.a;
    if (use_lighting && use_lighting_normals) {
        float3 v1 = normalize(in.v);
        float3 N1 = normalize(in.N) * (front_facing ? 1.0 : -1.0);
        a = correct_alpha(a, v1, N1);
    }

    MeshFragOut out;
    out.color = use_depth_prepass ? float4(0.0) : float4(Ifrag, a);
    return out;
}
