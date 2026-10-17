#version 460
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
layout(set = 1, binding = 0) uniform sampler2D quilt;
layout(set = 0, binding = 0, std140) uniform Calibration {
    vec4 pitch;
    vec4 rgb_pitch;
    vec4 tile;
    vec4 region_aspect;
} u;
vec2 tile_uv(vec2 uv, float z) {
    float t = floor(mod(z, 1.0) * u.tile.z);
    vec2 q = vec2((mod(t, u.tile.x) + uv.x) / u.tile.x,
                  (floor(t / u.tile.x) + uv.y) / u.tile.y) * u.region_aspect.xy;
    return vec2(q.x, 1.0 - q.y);
}
void main() {
    // Calibration uses display coordinates with the origin at bottom left.
    vec2 screen = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 uv = u.region_aspect.zw * (screen - 0.5) + 0.5;
    float z = dot(screen, u.pitch.xy) - u.pitch.z;
    color = vec4(texture(quilt, tile_uv(uv, z + u.rgb_pitch.r)).r,
                 texture(quilt, tile_uv(uv, z + u.rgb_pitch.g)).g,
                 texture(quilt, tile_uv(uv, z + u.rgb_pitch.b)).b, 1.0);
}
