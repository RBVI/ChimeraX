// Cross-bilateral denoiser for path-tracer output. 7x7 kernel, weighted
// by spatial Gaussian and depth similarity so silhouettes between atoms
// stay sharp while Monte-Carlo speckle inside a single atom's surface
// gets averaged across its neighbourhood. The depth buffer used as the
// edge stop is the rasterised impostor sphere depth (written by the
// OpaquePass before path tracing), so it naturally aligns with the
// path-traced color since both reflect the same atom positions.
#version 460

layout(set = 0, binding = 0) uniform Params {
    vec2  texel_size;       // 1/width, 1/height
    float depth_threshold;  // depth difference that blocks the blur
    float _pad;
} u;

layout(set = 1, binding = 0) uniform sampler2D color_tex;
layout(set = 1, binding = 1) uniform sampler2D depth_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

void main() {
    vec3  center_color = texture(color_tex, v_uv).rgb;
    float center_depth = texture(depth_tex, v_uv).r;

    // Background pixels (far plane): pass through. The depth buffer
    // holds the cleared far value here so neighbour samples would all
    // be far too — blurring is a no-op but skipping is cheaper.
    if (center_depth >= 0.9999) {
        frag_color = vec4(center_color, 1.0);
        return;
    }

    vec3  sum  = vec3(0.0);
    float wsum = 0.0;

    // 3x3 with a tight spatial weight (e^-2*dist² puts corners at
    // ~1.8% of centre, edges at ~13.5%). Center carries ~60% of the
    // total weight — a gentle smoothing that takes the worst speckle
    // off without blurring out indirect-lighting detail.
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 sample_uv = v_uv + vec2(dx, dy) * u.texel_size;

            vec3  sc = texture(color_tex, sample_uv).rgb;
            float sd = texture(depth_tex, sample_uv).r;

            float dist2 = float(dx * dx + dy * dy);
            float spatial_w = exp(-dist2 * 2.0);

            float dd = abs(center_depth - sd);
            float depth_w = exp(-dd * dd /
                                (u.depth_threshold * u.depth_threshold));

            float w = spatial_w * depth_w;
            sum  += sc * w;
            wsum += w;
        }
    }

    frag_color = vec4(sum / max(wsum, 1e-6), 1.0);
}
