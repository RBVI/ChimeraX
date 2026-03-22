// ChimeraX Metal shaders — optimised for Apple Silicon TBDR.
//
// Design notes
// ------------
// Apple Silicon uses Tile-Based Deferred Rendering (TBDR).  The GPU
// accumulates draw calls for a tile (typically 32×32 pixels) entirely
// in fast on-chip tile memory before writing to DRAM.  Key implications:
//
// 1. loadAction=Clear and storeAction=DontCare for depth means the tile
//    memory is never read from or written to DRAM for the depth buffer —
//    this is set in the C++ renderer, not in the shaders.
//
// 2. The depth pre-pass (fragmentDepthOnly) discards the fragment before
//    any texture sampling or lighting work, so the TBDR visibility resolve
//    happens before the expensive main fragment pass.
//
// 3. [[early_fragment_tests]] on the main fragment shader lets the GPU
//    skip shading for fragments that would fail the depth test, using
//    the depth pre-pass result.
//
// 4. GPU instancing via [[instance_id]]: one draw call covers N copies
//    of a Drawing.  Buffer slot 4 carries a (N,4,4) float32 array of
//    column-major instance transforms.  If no instance buffer is bound
//    (instanceCount==1, nil buffer), the shader uses the modelMatrix
//    from the Uniforms struct.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared uniform block — must match Uniforms in metal_renderer.hpp exactly.
// float3 is padded to float4 in MSL struct layout (16-byte alignment).
// ---------------------------------------------------------------------------
struct Uniforms {
    float4x4 modelMatrix;        // 64 bytes
    float4x4 viewMatrix;         // 64
    float4x4 projectionMatrix;   // 64
    float4x4 normalMatrix;       // 64 (inverse-transpose of model)

    float3 cameraPosition;  float _p0; // 16
    float3 lightPosition;   float lightRadius; // 16
    float3 lightColor;      float lightIntensity; // 16
    float3 ambientColor;    float ambientIntensity; // 16
    float2 screenSize;      float nearPlane; float farPlane; // 16
};

// ---------------------------------------------------------------------------
// Triangle pipeline
// ---------------------------------------------------------------------------

struct TriVert {
    float4 clipPos   [[position]];
    float3 worldPos;
    float3 worldNorm;
    float4 color;
};

// Vertex shader — supports both non-instanced (instanceBuf = nil) and
// instanced (instanceBuf = N×float4x4) rendering.
vertex TriVert vertexTriangle(
    uint                          vid            [[vertex_id]],
    uint                          iid            [[instance_id]],
    device const float3*          positions      [[buffer(0)]],
    device const float4*          colors         [[buffer(1)]],
    device const float3*          normals        [[buffer(2)]],
    constant Uniforms&            uniforms       [[buffer(3)]],
    device const float4x4*        instanceXforms [[buffer(4), function_constant(false)]])
{
    // Per-instance model transform: use instanceXforms[iid] if the buffer is
    // bound, otherwise fall through to uniforms.modelMatrix.
    // Because Metal does not allow conditional buffer access in a single PSO,
    // we use two specialisation-constant variants compiled at pipeline creation.
    // For simplicity here we rely on buffer(4) being nil-safe on Apple Silicon
    // (accessing a nil buffer reads zeros → identity column 0 == (1,0,0,0) etc.
    // is NOT guaranteed).  In practice the C++ renderer only binds buffer 4
    // when instanceBuf != nil, so the cases below are:
    //   instanceBuf == nil  → iid == 0, instanceXforms ptr is nil → crash risk.
    //   instanceBuf != nil  → instanceXforms[iid] is valid.
    //
    // Solution: the renderer always binds a 1-element identity buffer when
    // instanceBuf is nil and instanceCount == 1.  See addTriangles() below.
    // We accept this small overhead to keep a single PSO.
    float4x4 M = instanceXforms[iid];

    float4 worldPos4 = M * float4(positions[vid], 1.0);
    float3 worldNorm = normalize((M * float4(normals[vid], 0.0)).xyz);

    TriVert out;
    out.clipPos   = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos4;
    out.worldPos  = worldPos4.xyz;
    out.worldNorm = worldNorm;
    out.color     = colors[vid];
    return out;
}

// Fragment shader — Blinn-Phong with early fragment test (TBDR optimisation).
[[early_fragment_tests]]
fragment float4 fragmentTriangle(
    TriVert            in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    float3 N = normalize(in.worldNorm);
    float3 L = normalize(uniforms.lightPosition - in.worldPos);
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 H = normalize(L + V);

    float diff  = max(dot(N, L), 0.0h) * uniforms.lightIntensity;
    float spec  = pow(max(dot(N, H), 0.0h), 32.0h) * uniforms.lightIntensity * 0.3h;

    float3 ambient  = uniforms.ambientColor * uniforms.ambientIntensity;
    float3 diffuse  = uniforms.lightColor * diff;
    float3 specular = uniforms.lightColor * spec;

    float3 lit = in.color.rgb * (ambient + diffuse) + specular;
    return float4(saturate(lit), in.color.a);
}

// ---------------------------------------------------------------------------
// Depth-only pre-pass — no colour output, no fragment work.
// Apple Silicon TBDR uses this to fill tile depth before the main pass,
// allowing the GPU to discard occluded fragments in fragmentTriangle without
// ever running the lighting code.
// ---------------------------------------------------------------------------

vertex float4 vertexDepthOnly(
    uint                   vid            [[vertex_id]],
    uint                   iid            [[instance_id]],
    device const float3*   positions      [[buffer(0)]],
    device const float4x4* instanceXforms [[buffer(4)]],
    constant Uniforms&     uniforms       [[buffer(3)]])
{
    float4x4 M = instanceXforms[iid];
    return uniforms.projectionMatrix * uniforms.viewMatrix * M * float4(positions[vid], 1.0);
}

// Trivial fragment function required by Metal API even for depth-only PSO.
// The C++ pipeline uses MTLPixelFormatInvalid for the colour attachment so
// this fragment function is never actually invoked (raster ops drop it).
fragment void fragmentDepthOnly() {}

// ---------------------------------------------------------------------------
// Sphere imposter — raycast in fragment shader for per-pixel normals.
// Atoms in ball-and-stick / space-fill mode use this for accurate shading
// without a tessellated mesh.
// ---------------------------------------------------------------------------

struct SphereVert {
    float4 clipPos  [[position]];
    float3 center;
    float  radius;
    float4 color;
    float2 quad;    // billboard UV in [-1,1]
};

vertex SphereVert vertexSphere(
    uint                   vid      [[vertex_id]],
    device const float3*   centers  [[buffer(0)]],
    device const float4*   colors   [[buffer(1)]],
    device const float*    radii    [[buffer(2)]],
    constant Uniforms&     uniforms [[buffer(3)]])
{
    uint atomID  = vid >> 2;          // 4 vertices per sphere quad
    uint cornerID = vid & 3;

    float2 corners[4] = { {-1, 1}, {1, 1}, {-1,-1}, {1,-1} };
    float2 uv = corners[cornerID];

    float r = radii[atomID];
    float3 c = centers[atomID];

    // Expand billboard in view space so it always faces the camera.
    float4 viewC = uniforms.viewMatrix * float4(c, 1.0);
    viewC.xy += uv * r;

    SphereVert out;
    out.clipPos = uniforms.projectionMatrix * viewC;
    out.center  = c;
    out.radius  = r;
    out.color   = colors[atomID];
    out.quad    = uv;
    return out;
}

struct SphereFragOut {
    float4 color [[color(0)]];
    float  depth [[depth(less)]];  // write corrected depth for correct occlusion
};

[[early_fragment_tests]]
fragment SphereFragOut fragmentSphere(
    SphereVert         in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    // Discard corners outside the circle.
    float2 uv = in.quad;
    if (dot(uv, uv) > 1.0) discard_fragment();

    // Reconstruct surface normal from imposter UV.
    float z    = sqrt(1.0 - dot(uv, uv));
    float3 N   = float3(uv.x, uv.y, z);  // in view space
    float3 worldN = normalize((uniforms.viewMatrix.columns[0].xyz * N.x +
                               uniforms.viewMatrix.columns[1].xyz * N.y +
                               uniforms.viewMatrix.columns[2].xyz * N.z));

    // Corrected clip-space depth for the sphere surface point.
    float3 surfaceViewPos = (uniforms.viewMatrix * float4(in.center, 1.0)).xyz
                            + float3(uv, z) * in.radius;
    float4 surfaceClip    = uniforms.projectionMatrix * float4(surfaceViewPos, 1.0);
    float correctedDepth  = surfaceClip.z / surfaceClip.w;

    float3 L   = normalize(uniforms.lightPosition - in.center);
    float3 V   = normalize(uniforms.cameraPosition - in.center);
    float3 H   = normalize(L + V);
    float diff = max(dot(worldN, L), 0.0h) * uniforms.lightIntensity;
    float spec = pow(max(dot(worldN, H), 0.0h), 64.0h) * 0.4h;

    float3 lit = in.color.rgb * (uniforms.ambientColor * uniforms.ambientIntensity +
                                  uniforms.lightColor * diff) +
                 uniforms.lightColor * spec;

    SphereFragOut out;
    out.color = float4(saturate(lit), in.color.a);
    out.depth = correctedDepth;
    return out;
}

// ---------------------------------------------------------------------------
// Cylinder billboard — two quads per bond, expanded perpendicular to axis.
// ---------------------------------------------------------------------------

struct CylVert {
    float4 clipPos [[position]];
    float4 color;
    float3 worldPos;
    float3 axis;
};

vertex CylVert vertexCylinder(
    uint                   vid     [[vertex_id]],
    device const float3*   starts  [[buffer(0)]],
    device const float3*   ends    [[buffer(1)]],
    device const float4*   colors  [[buffer(2)]],
    device const float*    radii   [[buffer(3)]],
    constant Uniforms&     uniforms [[buffer(4)]])
{
    uint bondID  = vid >> 2;
    uint cornerID = vid & 3;

    float2 corners[4] = { {-1, 0}, {1, 0}, {-1, 1}, {1, 1} };
    float2 uv = corners[cornerID];

    float3 s    = starts[bondID];
    float3 e    = ends[bondID];
    float3 axis = e - s;
    float  len  = length(axis);
    float3 axN  = axis / max(len, 1e-6h);

    // Perpendicular in world space (stable for any axis direction).
    float3 ref  = abs(axN.y) < 0.9h ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 perp = normalize(cross(axN, ref));

    float r = radii[bondID];
    float3 wpos = s + axN * (uv.y * len) + perp * (uv.x * r);

    CylVert out;
    out.clipPos  = uniforms.projectionMatrix * uniforms.viewMatrix * float4(wpos, 1.0);
    out.color    = colors[bondID];
    out.worldPos = wpos;
    out.axis     = axN;
    return out;
}

[[early_fragment_tests]]
fragment float4 fragmentCylinder(
    CylVert            in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    // Approximate normal as cross(axis, view) for simple shading.
    float3 V = normalize(uniforms.cameraPosition - in.worldPos);
    float3 N = normalize(cross(in.axis, cross(V, in.axis)));

    float3 L   = normalize(uniforms.lightPosition - in.worldPos);
    float diff = max(dot(N, L), 0.0h) * uniforms.lightIntensity;
    float3 lit = in.color.rgb * (uniforms.ambientColor * uniforms.ambientIntensity +
                                  uniforms.lightColor * diff);
    return float4(saturate(lit), in.color.a);
}
