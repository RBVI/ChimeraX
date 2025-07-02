// metal_scene.cpp
// Implementation of scene management for Metal rendering

#include "metal_scene.hpp"
#include "metal_context.hpp"
#include <iostream>
#include <algorithm>

namespace chimerax {
namespace graphics {

// MetalLight implementation
MetalLight::MetalLight()
    : _type(LightType::Point)
    , _position(simd::float3(0.0f, 5.0f, 5.0f))
    , _direction(simd::float3(0.0f, -1.0f, -1.0f))
    , _color(simd::float3(1.0f, 1.0f, 1.0f))
    , _intensity(1.0f)
    , _radius(50.0f)
{
}

MetalLight::~MetalLight()
{
}

// MetalCamera implementation
MetalCamera::MetalCamera()
    : _position(simd::float3(0.0f, 0.0f, 5.0f))
    , _target(simd::float3(0.0f, 0.0f, 0.0f))
    , _up(simd::float3(0.0f, 1.0f, 0.0f))
    , _fov(45.0f)
    , _aspectRatio(1.0f)
    , _nearPlane(0.1f)
    , _farPlane(1000.0f)
{
}

MetalCamera::~MetalCamera()
{
}

simd::float4x4 MetalCamera::viewMatrix() const
{
    // Calculate view matrix (look-at)
    simd::float3 forward = simd::normalize(_target - _position);
    simd::float3 right = simd::normalize(simd::cross(forward, _up));
    simd::float3 upActual = simd::cross(right, forward);
    
    simd::float4x4 viewMatrix;
    
    // First three columns are the right, up, and forward basis vectors
    viewMatrix.columns[0] = simd::float4(right.x, upActual.x, -forward.x, 0.0f);
    viewMatrix.columns[1] = simd::float4(right.y, upActual.y, -forward.y, 0.0f);
    viewMatrix.columns[2] = simd::float4(right.z, upActual.z, -forward.z, 0.0f);
    
    // Fourth column is translation
    viewMatrix.columns[3] = simd::float4(
        -simd::dot(right, _position),
        -simd::dot(upActual, _position),
        simd::dot(forward, _position),
        1.0f
    );
    
    return viewMatrix;
}

simd::float4x4 MetalCamera::projectionMatrix() const
{
    // Calculate projection matrix (perspective)
    float tanHalfFov = tan(_fov * 0.5f * M_PI / 180.0f);
    float zRange = _farPlane - _nearPlane;
    
    simd::float4x4 projMatrix;
    
    projMatrix.columns[0] = simd::float4(1.0f / (tanHalfFov * _aspectRatio), 0.0f, 0.0f, 0.0f);
    projMatrix.columns[1] = simd::float4(0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f);
    projMatrix.columns[2] = simd::float4(0.0f, 0.0f, -(_farPlane + _nearPlane) / zRange, -1.0f);
    projMatrix.columns[3] = simd::float4(0.0f, 0.0f, -2.0f * _farPlane * _nearPlane / zRange, 0.0f);
    
    return projMatrix;
}

// MetalScene implementation
MetalScene::MetalScene(MetalContext* context)
    : _context(context)
    , _camera(nullptr)
    , _backgroundColor(simd::float4(0.2f, 0.2f, 0.2f, 1.0f))
    , _ambientColor(simd::float3(0.1f, 0.1f, 0.1f))
    , _ambientIntensity(1.0f)
{
}

MetalScene::~MetalScene()
{
}

bool MetalScene::initialize()
{
    // Create default camera
    _camera = std::make_unique<MetalCamera>();
    
    // Create default light
    auto defaultLight = std::make_shared<MetalLight>();
    addLight(defaultLight);
    
    return true;
}

void MetalScene::addLight(std::shared_ptr<MetalLight> light)
{
    if (light) {
        _lights.push_back(light);
    }
}

void MetalScene::removeLight(std::shared_ptr<MetalLight> light)
{
    auto it = std::find(_lights.begin(), _lights.end(), light);
    if (it != _lights.end()) {
        _lights.erase(it);
    }
}

void MetalScene::clearLights()
{
    _lights.clear();
}

std::shared_ptr<MetalLight> MetalScene::mainLight() const
{
    if (_lights.empty()) {
        return nullptr;
    }
    
    return _lights[0];
}

} // namespace graphics
} // namespace chimerax
