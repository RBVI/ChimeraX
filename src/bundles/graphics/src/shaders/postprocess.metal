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

// postprocess.metal -- Fullscreen-quad shaders for depth-based silhouette
// outlines, texture-based selection outlines, depth-buffer visualization,
// and 2-D / 3-D texture blending for multichannel volume rendering.

#include "common.metal"

// ---------------------------------------------------------------------------
//  Shared fullscreen-quad vertex shader.
//  Input: position (clip space), tex_coord.
// ---------------------------------------------------------------------------
struct PostVertexIn {
    float3 position  [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct PostVertexOut {
    float4 position     [[position]];
    float2 tex_coord_2d;
};

vertex PostVertexOut post_vertex(PostVertexIn in [[stage_in]])
{
    PostVertexOut out;
    out.tex_coord_2d = in.tex_coord;
    out.position = float4(in.position, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
//  Uniforms specific to the various post-processing effects.
// ---------------------------------------------------------------------------
struct PostUniforms {
    float4 color;
    float2 step;        // texel size (for texture outline)
    float2 jump;        // depth jump threshold and near/far ratio (depth outline)
    float3 step3;       // dx, dy, thickness (depth outline)
    float  tex_coord_z; // z slice for 3-D texture blending
    float2 colormap_range;
    float2 _pad;
};

// ============================= DEPTH OUTLINE ==============================
// Silhouette edges based on depth discontinuities (the USE_DEPTH_OUTLINE
// path in the GLSL shader).

fragment float4 depth_outline_fragment(
    PostVertexOut in                        [[stage_in]],
    constant PostUniforms &u               [[buffer(0)]],
    texture2d<float> tex2d                 [[texture(0)]],
    sampler samp                           [[sampler(0)]])
{
    float d0 = tex2d.sample(samp, in.tex_coord_2d).r;
    float ds = d0;
    float thickness = u.step3.z;
    int r = int(thickness);
    float r2 = thickness * thickness;

    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            if (i * i + j * j <= int(r2) && !(i == 0 && j == 0)) {
                float2 offset = float2(float(i) * u.step3.x,
                                       float(j) * u.step3.y);
                ds = min(ds, tex2d.sample(samp, in.tex_coord_2d + offset).r);
            }
        }
    }

    float nf = u.jump.y;   // near / far ratio
    float nf1 = 1.0 - nf;
    if (nf * (d0 - ds) < u.jump.x * (1.0 - nf1 * ds) * (1.0 - nf1 * d0))
        discard_fragment();

    return u.color;
}

// =========================== TEXTURE OUTLINE ==============================
// Selection outlines drawn around non-zero regions of a mask texture (the
// USE_TEXTURE_OUTLINE path in the GLSL shader).

fragment float4 texture_outline_fragment(
    PostVertexOut in                        [[stage_in]],
    constant PostUniforms &u               [[buffer(0)]],
    texture2d<float> tex2d                 [[texture(0)]],
    sampler samp                           [[sampler(0)]])
{
    float m0 = tex2d.sample(samp, in.tex_coord_2d).r;
    if (m0 != 0.0)
        discard_fragment();

    float mn = tex2d.sample(samp, in.tex_coord_2d + float2( u.step.x, 0.0)).r;
    mn = max(mn, tex2d.sample(samp, in.tex_coord_2d + float2(-u.step.x, 0.0)).r);
    mn = max(mn, tex2d.sample(samp, in.tex_coord_2d + float2(0.0,  u.step.y)).r);
    mn = max(mn, tex2d.sample(samp, in.tex_coord_2d + float2(0.0, -u.step.y)).r);

    if (mn == 0.0)
        discard_fragment();

    return u.color;
}

// ========================= SHOW DEPTH BUFFER ==============================
// Visualization of the depth buffer as a greyscale image.

fragment float4 show_depth_fragment(
    PostVertexOut in                        [[stage_in]],
    texture2d<float> depth_texture         [[texture(0)]],
    sampler samp                           [[sampler(0)]])
{
    float c = depth_texture.sample(samp, in.tex_coord_2d).r;
    return float4(c, c, c, 1.0);
}

// ========================= BLEND TEXTURE 2D ===============================
// Multichannel volume rendering: blend a 2-D texture with a modulation colour
// and optional colourmap.

constant bool use_blend_colormap [[function_constant(0)]];

fragment float4 blend_texture_2d_fragment(
    PostVertexOut in                        [[stage_in]],
    constant PostUniforms &u               [[buffer(0)]],
    texture2d<float> tex2d                 [[texture(0)]],
    texture1d<float> colormap              [[texture(1)]],
    sampler samp                           [[sampler(0)]])
{
    if (use_blend_colormap) {
        float tvalue = (tex2d.sample(samp, in.tex_coord_2d).r
                        - u.colormap_range.x) * u.colormap_range.y;
        return colormap.sample(samp, tvalue) * u.color;
    } else {
        return tex2d.sample(samp, in.tex_coord_2d) * u.color;
    }
}

// ========================= BLEND TEXTURE 3D ===============================
// Same as 2-D blend but sampling from a 3-D texture at a fixed z slice.

fragment float4 blend_texture_3d_fragment(
    PostVertexOut in                        [[stage_in]],
    constant PostUniforms &u               [[buffer(0)]],
    texture3d<float> tex3d                 [[texture(0)]],
    texture1d<float> colormap              [[texture(1)]],
    sampler samp                           [[sampler(0)]])
{
    float3 coord = float3(in.tex_coord_2d, u.tex_coord_z);
    if (use_blend_colormap) {
        float tvalue = (tex3d.sample(samp, coord).r
                        - u.colormap_range.x) * u.colormap_range.y;
        return colormap.sample(samp, tvalue) * u.color;
    } else {
        return tex3d.sample(samp, coord) * u.color;
    }
}
