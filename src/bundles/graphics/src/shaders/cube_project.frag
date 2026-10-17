#version 460
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0, std140) uniform Projection {
    vec4 parameters; // dome flag, aspect correction x/y, unused
    vec4 background;
} projection;
layout(set = 1, binding = 0) uniform sampler2D face0;
layout(set = 1, binding = 1) uniform sampler2D face1;
layout(set = 1, binding = 2) uniform sampler2D face2;
layout(set = 1, binding = 3) uniform sampler2D face3;
layout(set = 1, binding = 4) uniform sampler2D face4;
layout(set = 1, binding = 5) uniform sampler2D face5;
const float PI = 3.141592653589793;
void main() {
    vec2 p = vec2(2.0 * v_uv.x - 1.0, 1.0 - 2.0 * v_uv.y);
    vec3 d;
    if (projection.parameters.x > 0.5) {
        p *= projection.parameters.yz;
        float radius = length(p);
        if (radius > 1.0) { color = projection.background; return; }
        float phi = 0.5 * PI * radius;
        d = vec3(radius > 0.0 ? p * (sin(phi) / radius) : vec2(0.0), -cos(phi));
    } else {
        float theta = p.x * PI;
        float phi = (1.0 - p.y) * 0.5 * PI;
        d = vec3(sin(theta) * sin(phi), cos(phi), -cos(theta) * sin(phi));
    }
    vec3 a = abs(d);
    vec2 uv;
    // Face frames match _cube_map_face_views. Vulkan rows run top to bottom.
    if (a.x >= a.y && a.x >= a.z) {
        uv = 0.5 * (vec2(d.x >= 0.0 ? -d.z : d.z, d.y) / a.x + 1.0);
        color = d.x >= 0.0 ? texture(face0, uv) : texture(face1, uv);
    } else if (a.y >= a.z) {
        uv = 0.5 * (vec2(d.x, d.y >= 0.0 ? -d.z : d.z) / a.y + 1.0);
        color = d.y >= 0.0 ? texture(face2, uv) : texture(face3, uv);
    } else {
        uv = 0.5 * (vec2(d.z >= 0.0 ? d.x : -d.x, d.y) / a.z + 1.0);
        color = d.z >= 0.0 ? texture(face4, uv) : texture(face5, uv);
    }
}
