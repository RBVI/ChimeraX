"""
Metal-based graphics implementation for ChimeraX with multi-GPU acceleration
"""

import sys
import os
import platform
import numpy
from chimerax.core.graphics import Graphics
from chimerax.core.logger import log_error, log_info

# Import C++ extension module for Metal - would be built with Cython
# We use a try/except here to gracefully handle platforms where Metal is not available
try:
    from ._metal import (
        MetalContext, 
        MetalRenderer, 
        MetalScene, 
        MetalResources,
        MetalView,
        MetalArgBuffer,
        MetalMultiGPU,
        # Exception type for Metal errors
        MetalError
    )
    _have_metal = True
except ImportError as e:
    _have_metal = False
    log_error(f"Failed to import Metal module: {e}")

# Check if we're on a platform that supports Metal
def is_metal_supported():
    """Check if the current platform supports Metal"""
    if not _have_metal:
        return False
    
    # Metal is only supported on macOS
    if platform.system() != "Darwin":
        return False
    
    # Check macOS version - Metal requires 10.14+ for all features we use
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        major, minor = map(int, mac_ver.split('.')[:2])
        if (major < 10) or (major == 10 and minor < 14):
            return False
    
    return True

# Constants for multi-GPU strategies
MULTI_GPU_STRATEGY_SPLIT_FRAME = 0
MULTI_GPU_STRATEGY_TASK_BASED = 1
MULTI_GPU_STRATEGY_ALTERNATING = 2
MULTI_GPU_STRATEGY_COMPUTE_OFFLOAD = 3

class MetalGraphics(Graphics):
    """Metal-based graphics implementation for ChimeraX with multi-GPU support"""
    
    def __init__(self, session, **kw):
        """Initialize Metal graphics"""
        super().__init__(session, **kw)
        
        self._metal_view = None
        self._multi_gpu = None
        self._multi_gpu_enabled = False
        self._multi_gpu_strategy = MULTI_GPU_STRATEGY_SPLIT_FRAME
        
        # Create Metal view if supported
        if is_metal_supported():
            try:
                # Create the low-level Metal view
                self._metal_view = MetalView()
                
                # Initialize multi-GPU manager if available
                self._multi_gpu = MetalMultiGPU()
                
                log_info("Using Metal graphics renderer with multi-GPU support")
            except Exception as e:
                log_error(f"Failed to create Metal renderer: {e}")
                self._metal_view = None
                self._multi_gpu = None
    
    def initialize(self, width, height, window_id, make_current=False):
        """Initialize the Metal rendering context"""
        if self._metal_view:
            try:
                success = self._metal_view.initialize(window_id, width, height)
                if not success:
                    log_error("Failed to initialize Metal view")
                    self._metal_view = None
                    self._multi_gpu = None
                    return super().initialize(width, height, window_id, make_current)
                
                # Initialize multi-GPU if available
                if self._multi_gpu:
                    # Get Metal context from view
                    metal_context = self._metal_view.context()
                    if metal_context:
                        self._multi_gpu.initialize(metal_context)
                
                # Successfully initialized Metal
                return True
            except Exception as e:
                log_error(f"Error initializing Metal view: {e}")
                self._metal_view = None
                self._multi_gpu = None
        
        # Fall back to OpenGL if Metal is not available
        return super().initialize(width, height, window_id, make_current)
    
    def make_current(self):
        """Make the Metal context current if using Metal, otherwise use OpenGL"""
        if self._metal_view:
            # Metal doesn't use the concept of "current context" like OpenGL
            # so this is a no-op for Metal
            return True
        return super().make_current()
    
    def done_current(self):
        """Release the current context if using OpenGL"""
        if not self._metal_view:
            return super().done_current()
    
    def swap_buffers(self):
        """Swap buffers or present the Metal drawable"""
        if self._metal_view:
            self._metal_view.render()
        else:
            super().swap_buffers()
    
    def render(self, drawing, camera, render_target=None):
        """Render the drawing using metal or fall back to OpenGL"""
        if self._metal_view:
            # Update scene with camera and drawing
            metal_scene = self._metal_view.scene()
            if metal_scene:
                # Update camera parameters
                metal_camera = metal_scene.camera()
                if metal_camera and camera:
                    # Convert camera position and orientation
                    eye_pos = camera.position.origin
                    metal_camera.setPosition((eye_pos[0], eye_pos[1], eye_pos[2]))
                    
                    # Set camera target (look-at point)
                    look_at = eye_pos + camera.view_direction() * 10.0
                    metal_camera.setTarget((look_at[0], look_at[1], look_at[2]))
                    
                    # Set camera up vector
                    up_vector = camera.position.z_axis
                    metal_camera.setUp((up_vector[0], up_vector[1], up_vector[2]))
                    
                    # Set perspective parameters
                    metal_camera.setFov(camera.field_of_view)
                    metal_camera.setNearPlane(camera.near_clip_distance)
                    metal_camera.setFarPlane(camera.far_clip_distance)
                
                # TODO: Convert drawing to Metal representation
                # This is a complex step that would require translating the
                # ChimeraX drawing structures into Metal-compatible buffers
            
            # Trigger rendering
            self._metal_view.render()
        else:
            super().render(drawing, camera, render_target)
    
    def enable_multi_gpu(self, enable=True, strategy=MULTI_GPU_STRATEGY_SPLIT_FRAME):
        """Enable or disable multi-GPU rendering with Metal"""
        if not self._metal_view or not self._multi_gpu:
            log_error("Cannot enable multi-GPU: Metal is not active or multi-GPU not supported")
            return False
        
        # Set the multi-GPU strategy
        if enable:
            self._multi_gpu_strategy = strategy
            success = self._multi_gpu.enable(True, strategy)
        else:
            success = self._multi_gpu.enable(False, MULTI_GPU_STRATEGY_SPLIT_FRAME)
            
        self._multi_gpu_enabled = enable and success
        
        # Update renderer settings if multi-GPU enabled
        if self._multi_gpu_enabled:
            # Get Metal renderer from view and configure it for multi-GPU
            renderer = self._metal_view.renderer()
            if renderer:
                renderer.setMultiGPUMode(True, strategy)
        
        return success
    
    def is_multi_gpu_enabled(self):
        """Check if multi-GPU rendering is enabled"""
        return self._metal_view is not None and self._multi_gpu is not None and self._multi_gpu_enabled
    
    def get_gpu_devices(self):
        """Get information about available Metal GPU devices"""
        if not self._metal_view:
            return []
            
        if self._multi_gpu:
            return self._multi_gpu.getDeviceInfo()
        
        # Fallback if multi-GPU manager is not available
        metal_context = self._metal_view.context()
        if metal_context:
            # Just return primary device info
            device_name = metal_context.deviceName()
            return [{"name": device_name, "is_primary": True, "unified_memory": metal_context.supportsUnifiedMemory()}]
            
        return []
    
    def begin_frame_capture(self):
        """Begin Metal frame capture for debugging"""
        if self._metal_view:
            self._metal_view.beginCapture()
    
    def end_frame_capture(self):
        """End Metal frame capture"""
        if self._metal_view:
            self._metal_view.endCapture()
    
    def resize(self, width, height):
        """Resize the rendering view"""
        if self._metal_view:
            self._metal_view.resize(width, height)
        else:
            super().resize(width, height)
    
    def set_background_color(self, color):
        """Set the background color for rendering"""
        if self._metal_view:
            scene = self._metal_view.scene()
            if scene:
                # Convert RGBA color (0-1 range)
                metal_color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
                scene.setBackgroundColor(metal_color)
        else:
            super().set_background_color(color)
    
    def get_capabilities(self):
        """Return dictionary of supported capabilities"""
        capabilities = super().get_capabilities()
        
        if self._metal_view:
            # Add Metal-specific capabilities
            metal_capabilities = {
                "api": "Metal",
                "multi_gpu": self._multi_gpu is not None,
                "ray_tracing": True,  # Metal 3 supports ray tracing
                "mesh_shaders": True, # Metal 3 supports mesh shaders
                "indirect_drawing": True,
                "argument_buffers": True,
                "unified_memory": False
            }
            
            # Check for unified memory (Apple Silicon)
            metal_context = self._metal_view.context()
            if metal_context:
                metal_capabilities["unified_memory"] = metal_context.supportsUnifiedMemory()
            
            capabilities.update(metal_capabilities)
        
        return capabilities
