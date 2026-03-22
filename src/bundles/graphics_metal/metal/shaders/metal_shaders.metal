// ChimeraX Metal shaders
// All geometry is stored as fp32; uniforms use simd_float4x4.
//
// Buffer index conventions (match metal_renderer.hpp Uniforms struct):
//   vertex buffer 0 : float3 positions
//   vertex buffer 1 : float4 colors (RGBA 0-1)
//   vertex buffer 2 : float3 normals
//   vertex buffer 3 : Uniforms (both vertex and fragment stages)

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared uniform block — must match Uniforms in metal_renderer.hpp
// ---------------------------------------------------------------------------
struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 normalMatrix;

    float3 cameraPosition;
    float  _pad0;

    float3 lightPosition;
    float  lightRadius;
    float3 lightColor;
    float  lightIntensity;
    float3 ambientColor;
    float  ambientIntensity;
};

// ---------------------------------------------------------------------------
// Triangle pipeline
// ---------------------------------------------------------------------------

struct TriangleVertexIn {
    float3 position [[attribute(0)]];
    float4 color    [[attribute(1)]];
    float3 normal   [[attribute(2)]];
};

struct TriangleVertexOut {
    float4 position [[position]];
    float4 color;
    float3 worldNormal;
    float3 worldPosition;
};

vertex TriangleVertexOut vertexTriangle(
    uint           vid       [[vertex_id]],
    device const float3* positions [[buffer(0)]],
    device const float4* colors    [[buffer(1)]],
    device const float3* normals   [[buffer(2)]],
    constant Uniforms&  uniforms   [[buffer(3)]])
{
    TriangleVertexOut out;
    float4 worldPos = uniforms.modelMatrix * float4(positions[vid], 1.0);
    out.position    = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.worldPosition = worldPos.xyz;
    out.worldNormal   = normalize((uniforms.normalMatrix * float4(normals[vid], 0.0)).xyz);
    out.color         = colors[vid];
    return out;
}

fragment float4 fragmentTriangle(
    TriangleVertexOut  in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    float3 N = normalize(in.worldNormal);
    float3 L = normalize(uniforms.lightPosition - in.worldPosition);

    // Blinn-Phong diffuse + ambient
    float diff    = max(dot(N, L), 0.0) * uniforms.lightIntensity;
    float3 ambient = uniforms.ambientColor * uniforms.ambientIntensity;
    float3 diffuse = uniforms.lightColor * diff;

    float3 litColor = in.color.rgb * (ambient + diffuse);
    return float4(litColor, in.color.a);
}

// ---------------------------------------------------------------------------
// Sphere imposter pipeline (raycast sphere in fragment shader)
// ---------------------------------------------------------------------------

struct SphereVertexOut {
    float4 position   [[position]];
    float3 center;
    float  radius;
    float4 color;
};

vertex SphereVertexOut vertexSphere(
    uint                vid       [[vertex_id]],
    device const float3* centers  [[buffer(0)]],
    device const float4* colors   [[buffer(1)]],
    device const float*  radii    [[buffer(2)]],
    constant Uniforms&   uniforms  [[buffer(3)]])
{
    SphereVertexOut out;
    float r = radii[vid];
    float3 c = centers[vid];

    // Expand billboard quad in view space.
    // vid within each quad: 0=TL 1=TR 2=BL 3=BR (using triangle strip)
    float2 corners[4] = { {-1, 1}, {1, 1}, {-1,-1}, {1,-1} };
    float2 corner = corners[vid & 3] * r;

    float4 viewCenter = uniforms.viewMatrix * float4(c, 1.0);
    viewCenter.xy += corner;

    out.position = uniforms.projectionMatrix * viewCenter;
    out.center   = c;
    out.radius   = r;
    out.color    = colors[vid >> 2];
    return out;
}

fragment float4 fragmentSphere(
    SphereVertexOut    in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    // Simple per-fragment lighting using center as normal proxy.
    float3 L = normalize(uniforms.lightPosition - in.center);
    float diff = max(dot(float3(0, 0, 1), L), 0.0) * uniforms.lightIntensity;
    float3 lit = in.color.rgb * (uniforms.ambientColor * uniforms.ambientIntensity
                                 + uniforms.lightColor * diff);
    return float4(lit, in.color.a);
}

// ---------------------------------------------------------------------------
// Cylinder pipeline (billboard imposter)
// ---------------------------------------------------------------------------

struct CylinderVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex CylinderVertexOut vertexCylinder(
    uint                vid       [[vertex_id]],
    device const float3* starts   [[buffer(0)]],
    device const float3* ends     [[buffer(1)]],
    device const float4* colors   [[buffer(2)]],
    device const float*  radii    [[buffer(3)]],
    constant Uniforms&   uniforms  [[buffer(4)]])
{
    CylinderVertexOut out;
    float3 s = starts[vid >> 2];
    float3 e = ends[vid >> 2];
    float r   = radii[vid >> 2];

    float2 corners[4] = { {-1, 0}, {1, 0}, {-1, 1}, {1, 1} };
    float2 corner = corners[vid & 3];

    float3 axis  = normalize(e - s);
    float3 perp  = normalize(cross(axis, float3(0, 1, 0.001)));
    float3 wpos  = s + axis * (corner.y * length(e - s)) + perp * (corner.x * r);

    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * float4(wpos, 1.0);
    out.color    = colors[vid >> 2];
    return out;
}

fragment float4 fragmentCylinder(
    CylinderVertexOut  in       [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]])
{
    return in.color;
}
