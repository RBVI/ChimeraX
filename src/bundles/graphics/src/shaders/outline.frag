// Highlight-outline composite. The mask texture holds the highlighted
// geometry rendered all-white (depth-tested against the scene depth, so
// only visible parts are masked). An outline pixel is one whose own mask
// red is zero but that has a masked pixel within `step` in one of the
// four axis directions — the same neighborhood rule the OpenGL backend's
// TEXTURE_OUTLINE shader uses. Everything else is discarded so the
// underlying scene shows through.
#version 460

layout(set = 0, binding = 0) uniform OutlineParams {
    vec4 color;     // outline color (rgba)
    vec4 step_pad;  // .xy = pixel_width / mask size
} u;

layout(set = 1, binding = 0) uniform sampler2D mask_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    if (texture(mask_tex, v_uv).r > 0.0)
        discard;
    vec2 dx = vec2(u.step_pad.x, 0.0);
    vec2 dy = vec2(0.0, u.step_pad.y);
    float neighbors = texture(mask_tex, v_uv + dx).r
                    + texture(mask_tex, v_uv - dx).r
                    + texture(mask_tex, v_uv + dy).r
                    + texture(mask_tex, v_uv - dy).r;
    if (neighbors <= 0.0)
        discard;
    frag_color = u.color;
}
