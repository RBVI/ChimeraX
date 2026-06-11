// Depth-edge silhouettes. Port of the OpenGL backend's USE_DEPTH_OUTLINE
// shader (fragmentShader.txt): for each pixel, take the minimum depth in
// a disc of radius `thickness`; where the center depth jumps past that
// minimum by more than `depth_jump` (perspective-corrected by the
// near/far ratio), draw the silhouette color, otherwise discard so the
// scene shows through. The depth texture is a copy of the scene depth
// made by vulkan_copy_depth_texture.
#version 460

layout(set = 0, binding = 0) uniform SilhouetteParams {
    vec4 color;          // silhouette color (rgba)
    vec4 jump_nf_step;   // .x = depth_jump, .y = near/far ratio,
                         // .zw = 1/width, 1/height
    vec4 thickness_pad;  // .x = thickness in pixels
} u;

layout(set = 1, binding = 0) uniform sampler2D depth_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    float d0 = texture(depth_tex, v_uv).r;
    float ds = d0;
    float thickness = u.thickness_pad.x;
    int r = int(thickness);
    float r2 = thickness * thickness;
    for (int i = -r; i <= r; ++i)
        for (int j = -r; j <= r; ++j)
            if (i * i + j * j <= r2 && !(i == 0 && j == 0))
                ds = min(ds, texture(depth_tex,
                    v_uv + vec2(float(i) * u.jump_nf_step.z,
                                float(j) * u.jump_nf_step.w)).r);
    // Correct depth compression based on perspective near/far ratio.
    float nf = u.jump_nf_step.y;
    float nf1 = 1.0 - nf;
    if (nf * (d0 - ds) < u.jump_nf_step.x * (1.0 - nf1 * ds) * (1.0 - nf1 * d0))
        discard;
    frag_color = u.color;
}
