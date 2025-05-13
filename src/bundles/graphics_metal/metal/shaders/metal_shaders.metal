// metal_shaders.metal
// Metal shaders for rendering molecular structures in ChimeraX

#include <metal_stdlib>
using namespace metal;

// Structures that match C++ counterparts
struct Uniforms {
    // Matrices
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 normalMatrix;
    
    // Camera
    float3 cameraPosition;
    float padding1;
    
    // Lighting
    float3 lightPosition;
    float lightRadius;
    float3 lightColor;
    float lightIntensity;
    float3 ambientColor;
    float ambientIntensity;
};

struct MaterialProperties {
    // Basic properties
    float4 color;
    float roughness;
    float metallic;
    float ambientOcclusion;
    float padding;
    
    // Additional properties for molecular rendering
    float atomRadius;
    float bondRadius;
    float outlineWidth;
    float outlineStrength;
};

// Vertex shader outputs
struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float4 color;
    float2 texCoord;
};

// Utility functions
float3 calculateNormal(float3 position, float3 center, float radius) {
    return normalize(position - center);
}

float3 calculatePhongLighting(float3 worldPos, float3 normal, float3 viewDir, float3 diffuseColor, constant Uniforms& uniforms) {
    // Calculate light direction and attenuation
    float3 lightDir = normalize(uniforms.lightPosition - worldPos);
    float distance = length(uniforms.lightPosition - worldPos);
    float attenuation = 1.0 / (1.0 + distance * distance / (uniforms.lightRadius * uniforms.lightRadius));
    
    // Ambient component
    float3 ambient = uniforms.ambientColor * uniforms.ambientIntensity * diffuseColor;
    
    // Diffuse component
    float NdotL = max(dot(normal, lightDir), 0.0);
    float3 diffuse = uniforms.lightColor * NdotL * diffuseColor * uniforms.lightIntensity;
    
    // Specular component
    float3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    float3 specular = uniforms.lightColor * spec * 0.5 * uniforms.lightIntensity;
    
    // Combine lighting components with attenuation
    return ambient + (diffuse + specular) * attenuation;
}

// Sphere rendering
vertex VertexOut vertexSphere(uint vertexID [[vertex_id]],
                              constant float3* positions [[buffer(0)]],
                              constant float4* colors [[buffer(1)]],
                              constant float* radii [[buffer(2)]],
                              constant Uniforms& uniforms [[buffer(3)]]) {
    // Get sphere data
    float3 center = positions[vertexID];
    float4 color = colors[vertexID];
    float radius = radii[vertexID];
    
    // Transform center position
    float4 worldPos = uniforms.modelMatrix * float4(center, 1.0);
    
    // For point sprites, we just pass the center
    VertexOut out;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.worldPosition = worldPos.xyz;
    out.normal = float3(0, 0, 1); // Will be calculated in fragment shader
    out.color = color;
    out.texCoord = float2(0, 0);
    
    // Adjust point size based on radius and perspective
    // Note: Metal doesn't support gl_PointSize directly, so this will need
    // to be implemented differently in a real application
    
    return out;
}

fragment float4 fragmentSphere(VertexOut in [[stage_in]],
                              constant Uniforms& uniforms [[buffer(0)]]) {
    // Calculate normal based on sphere equation
    float3 normal = normalize(in.normal);
    
    // Calculate view direction
    float3 viewDir = normalize(uniforms.cameraPosition - in.worldPosition);
    
    // Calculate lighting
    float3 litColor = calculatePhongLighting(in.worldPosition, normal, viewDir, in.color.rgb, uniforms);
    
    return float4(litColor, in.color.a);
}

// Cylinder rendering
vertex VertexOut vertexCylinder(uint vertexID [[vertex_id]],
                               constant float3* startPositions [[buffer(0)]],
                               constant float3* endPositions [[buffer(1)]],
                               constant float4* colors [[buffer(2)]],
                               constant float* radii [[buffer(3)]],
                               constant Uniforms& uniforms [[buffer(4)]]) {
    // Calculate which end of the cylinder this vertex represents
    uint index = vertexID / 2;
    bool isStart = (vertexID % 2) == 0;
    
    // Get cylinder data
    float3 startPos = startPositions[index];
    float3 endPos = endPositions[index];
    float4 color = colors[index];
    float radius = radii[index];
    
    // Position is either start or end
    float3 position = isStart ? startPos : endPos;
    
    // Calculate cylinder direction
    float3 direction = normalize(endPos - startPos);
    
    // Transform position
    float4 worldPos = uniforms.modelMatrix * float4(position, 1.0);
    
    // For line primitives, we just pass the end points
    // The actual cylinder geometry would be generated in a geometry shader
    // or by using instanced rendering with a cylinder mesh
    VertexOut out;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.worldPosition = worldPos.xyz;
    out.normal = float3(0, 0, 1); // Placeholder
    out.color = color;
    out.texCoord = float2(0, 0);
    
    return out;
}

fragment float4 fragmentCylinder(VertexOut in [[stage_in]],
                                constant Uniforms& uniforms [[buffer(0)]]) {
    // This is a simplified version - a real implementation would
    // calculate normals and lighting for a cylinder
    
    // Calculate view direction
    float3 viewDir = normalize(uniforms.cameraPosition - in.worldPosition);
    
    // Calculate lighting
    float3 litColor = calculatePhongLighting(in.worldPosition, in.normal, viewDir, in.color.rgb, uniforms);
    
    return float4(litColor, in.color.a);
}

// Triangle mesh rendering
vertex VertexOut vertexTriangle(uint vertexID [[vertex_id]],
                               constant float3* positions [[buffer(0)]],
                               constant float4* colors [[buffer(1)]],
                               constant float3* normals [[buffer(2)]],
                               constant Uniforms& uniforms [[buffer(3)]]) {
    float3 position = positions[vertexID];
    float4 color = colors[vertexID];
    float3 normal = normals[vertexID];
    
    // Transform position to world space
    float4 worldPos = uniforms.modelMatrix * float4(position, 1.0);
    
    // Transform normal to world space
    float3 worldNormal = (uniforms.normalMatrix * float4(normal, 0.0)).xyz;
    
    VertexOut out;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.worldPosition = worldPos.xyz;
    out.normal = normalize(worldNormal);
    out.color = color;
    out.texCoord = float2(0, 0);
    
    return out;
}

fragment float4 fragmentTriangle(VertexOut in [[stage_in]],
                                constant Uniforms& uniforms [[buffer(0)]]) {
    // Calculate view direction
    float3 viewDir = normalize(uniforms.cameraPosition - in.worldPosition);
    
    // Calculate lighting
    float3 litColor = calculatePhongLighting(in.worldPosition, in.normal, viewDir, in.color.rgb, uniforms);
    
    return float4(litColor, in.color.a);
}
