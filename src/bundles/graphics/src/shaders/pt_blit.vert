// Fullscreen quad vertex shader for the path-tracer blit pass.
#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_texcoord;

layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = a_texcoord.xy;
    gl_Position = vec4(a_position, 1.0);
}
