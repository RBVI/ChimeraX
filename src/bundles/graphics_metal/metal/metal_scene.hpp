// metal_scene.hpp
// Scene management for Metal rendering in ChimeraX

#pragma once

#import <Metal/Metal.h>
#include <simd/simd.h>
#include <memory>
#include <vector>
#include <string>

namespace chimerax {
namespace graphics {

// Forward declarations
class MetalContext;
class MetalCamera;

/**
 * Light types supported by the Metal renderer
 */
enum class LightType {
    Directional,
    Point,
    Spot
};

/**
 * Light source in the scene
 */
class MetalLight {
public:
    MetalLight();
    ~MetalLight();
    
    LightType type() const { return _type; }
    void setType(LightType type) { _type = type; }
    
    simd::float3 position() const { return _position; }
    void setPosition(simd::float3 position) { _position = position; }
    
    simd::float3 direction() const { return _direction; }
    void setDirection(simd::float3 direction) { _direction = direction; }
    
    simd::float3 color() const { return _color; }
    void setColor(simd::float3 color) { _color = color; }
    
    float intensity() const { return _intensity; }
    void setIntensity(float intensity) { _intensity = intensity; }
    
    float radius() const { return _radius; }
    void setRadius(float radius) { _radius = radius; }
    
private:
    LightType _type;
    simd::float3 _position;
    simd::float3 _direction;
    simd::float3 _color;
    float _intensity;
    float _radius;
};

/**
 * Camera for the Metal renderer
 */
class MetalCamera {
public:
    MetalCamera();
    ~MetalCamera();
    
    simd::float3 position() const { return _position; }
    void setPosition(simd::float3 position) { _position = position; }
    
    simd::float3 target() const { return _target; }
    void setTarget(simd::float3 target) { _target = target; }
    
    simd::float3 up() const { return _up; }
    void setUp(simd::float3 up) { _up = up; }
    
    float fov() const { return _fov; }
    void setFov(float fov) { _fov = fov; }
    
    float aspectRatio() const { return _aspectRatio; }
    void setAspectRatio(float aspectRatio) { _aspectRatio = aspectRatio; }
    
    float nearPlane() const { return _nearPlane; }
    void setNearPlane(float nearPlane) { _nearPlane = nearPlane; }
    
    float farPlane() const { return _farPlane; }
    void setFarPlane(float farPlane) { _farPlane = farPlane; }
    
    simd::float4x4 viewMatrix() const;
    simd::float4x4 projectionMatrix() const;
    
private:
    simd::float3 _position;
    simd::float3 _target;
    simd::float3 _up;
    float _fov;
    float _aspectRatio;
    float _nearPlane;
    float _farPlane;
};

/**
 * Scene management for Metal rendering
 */
class MetalScene {
public:
    MetalScene(MetalContext* context);
    ~MetalScene();
    
    // Initialization
    bool initialize();
    
    // Camera access
    MetalCamera* camera() const { return _camera.get(); }
    
    // Light management
    void addLight(std::shared_ptr<MetalLight> light);
    void removeLight(std::shared_ptr<MetalLight> light);
    void clearLights();
    std::shared_ptr<MetalLight> mainLight() const;
    
    // Background
    simd::float4 backgroundColor() const { return _backgroundColor; }
    void setBackgroundColor(simd::float4 color) { _backgroundColor = color; }
    
    // Ambient lighting
    simd::float3 ambientColor() const { return _ambientColor; }
    void setAmbientColor(simd::float3 color) { _ambientColor = color; }
    
    float ambientIntensity() const { return _ambientIntensity; }
    void setAmbientIntensity(float intensity) { _ambientIntensity = intensity; }
    
private:
    MetalContext* _context;
    std::unique_ptr<MetalCamera> _camera;
    std::vector<std::shared_ptr<MetalLight>> _lights;
    simd::float4 _backgroundColor;
    simd::float3 _ambientColor;
    float _ambientIntensity;
};

} // namespace graphics
} // namespace chimerax
