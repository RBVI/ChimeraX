// RTAO composite fragment shader.
//
// Reads the AO texture (ao_tex.r in [0,1], 1 = fully exposed) and outputs
// black with alpha = (1 - ao). Combined with the standard SRC_ALPHA over
// blend the renderer uses, this multiplies the existing framebuffer color
// by the AO factor — i.e. `dst' = dst * ao`.
#version 460

layout(set = 1, binding = 0) uniform sampler2D ao_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    float ao = texture(ao_tex, v_uv).r;
    frag_color = vec4(0.0, 0.0, 0.0, 1.0 - ao);
}
