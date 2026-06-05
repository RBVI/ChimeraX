// OIT composite - blends the WBOIT accumulation/revealage textures
// over the opaque scene using standard alpha blending.
//
// Pipeline blend: srcAlpha, oneMinusSrcAlpha (standard alpha).
// Output alpha = (1 - revealage) = fraction of light absorbed by
// transparent layers; color = average weighted transparent color.
#version 460

layout(set = 1, binding = 0) uniform sampler2D accum_tex;
layout(set = 1, binding = 1) uniform sampler2D reveal_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    vec4 accum = texture(accum_tex, v_uv);
    float revealage = texture(reveal_tex, v_uv).r;

    // If nothing was drawn (revealage == 1), discard to avoid
    // blending a zero-color transparent layer over the opaque scene.
    if (revealage >= 0.999)
        discard;

    // Reconstruct average transparent color.
    vec3 avg_color = accum.rgb / max(accum.a, 1e-5);

    // Output with alpha = absorption fraction.  Standard SRC_ALPHA
    // blending gives: dst' = avg_color * alpha + opaque * (1 - alpha)
    //               = avg_color * (1-revealage) + opaque * revealage.
    frag_color = vec4(avg_color, 1.0 - revealage);
}
