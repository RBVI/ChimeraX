// Fullscreen blit of the path-tracer color texture to the framebuffer.
#version 460

layout(set = 1, binding = 0) uniform sampler2D color_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = texture(color_tex, v_uv);
}
