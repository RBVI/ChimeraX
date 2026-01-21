"""
Cython wrapper for Metal C++ classes
"""

# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr

# Define Metal exception class
class MetalError(Exception):
    """Exception raised for Metal-related errors."""
    pass

# Forward declarations of C++ classes
cdef extern from "metal_context.hpp" namespace "chimerax::graphics_metal":
    cdef cppclass MetalContext:
        MetalContext() except +
        bool initialize() except +
        bool isInitialized() except +
        string deviceName() except +
        string deviceVendor() except +
        bool supportsUnifiedMemory() except +
        bool supportsRayTracing() except +
        bool supportsMeshShaders() except +
        void beginCapture() except +
        void endCapture() except +

cdef extern from "metal_scene.hpp" namespace "chimerax::graphics_metal":
    cdef cppclass MetalCamera:
        MetalCamera() except +
        void setPosition(float, float, float) except +
        void setTarget(float, float, float) except +
        void setUp(float, float, float) except +
        void setFov(float) except +
        void setAspectRatio(float) except +
        void setNearPlane(float) except +
        void setFarPlane(float) except +

    cdef cppclass MetalScene:
        MetalScene(MetalContext*) except +
        bool initialize() except +
        MetalCamera* camera() except +
        void setBackgroundColor(float, float, float, float) except +
        void setAmbientColor(float, float, float) except +
        void setAmbientIntensity(float) except +

cdef extern from "metal_argbuffer_manager.hpp" namespace "chimerax::graphics_metal":
    cdef cppclass MetalArgBuffer:
        MetalArgBuffer() except +
        string name() except +

cdef extern from "metal_multi_gpu.hpp" namespace "chimerax::graphics_metal":
    cdef enum MultiGPUStrategy:
        SplitFrame "chimerax::graphics_metal::MultiGPUStrategy::SplitFrame"
        TaskBased "chimerax::graphics_metal::MultiGPUStrategy::TaskBased"
        Alternating "chimerax::graphics_metal::MultiGPUStrategy::Alternating"
        ComputeOffload "chimerax::graphics_metal::MultiGPUStrategy::ComputeOffload"
        
    cdef struct GPUDeviceInfo:
        string name
        bool isPrimary
        bool isActive
        bool unifiedMemory
        unsigned long long memorySize
    
    cdef cppclass MetalMultiGPU:
        MetalMultiGPU() except +
        bool initialize(MetalContext*) except +
        vector[GPUDeviceInfo] getDeviceInfo() except +
        bool enable(bool, MultiGPUStrategy) except +
        bool isEnabled() except +
        MultiGPUStrategy getStrategy() except +

cdef extern from "metal_renderer.hpp" namespace "chimerax::graphics_metal":
    cdef cppclass MetalRenderer:
        MetalRenderer(MetalContext*) except +
        bool initialize() except +
        void setScene(MetalScene*) except +
        void beginFrame() except +
        void endFrame() except +
        void setMultiGPUMode(bool, int) except +

cdef extern from "metal_resources.hpp" namespace "chimerax::graphics_metal":
    cdef cppclass MetalResources:
        MetalResources(MetalContext*) except +
        bool initialize() except +

# PyMetalContext - Wrapper for MetalContext
cdef class PyMetalContext:
    cdef MetalContext* _context
    
    def __cinit__(self):
        self._context = new MetalContext()
        if self._context == NULL:
            raise MemoryError("Failed to allocate MetalContext")
    
    def __dealloc__(self):
        if self._context != NULL:
            del self._context
            self._context = NULL
    
    def initialize(self):
        """Initialize the Metal context"""
        if not self._context.initialize():
            raise MetalError("Failed to initialize Metal context")
        return True
        
    def isInitialized(self):
        """Check if the Metal context is initialized"""
        return self._context.isInitialized()
        
    def deviceName(self):
        """Get the name of the Metal device"""
        return self._context.deviceName().decode('utf-8')
        
    def deviceVendor(self):
        """Get the vendor of the Metal device"""
        return self._context.deviceVendor().decode('utf-8')
        
    def supportsUnifiedMemory(self):
        """Check if the device supports unified memory (Apple Silicon)"""
        return self._context.supportsUnifiedMemory()
        
    def supportsRayTracing(self):
        """Check if the device supports ray tracing"""
        return self._context.supportsRayTracing()
        
    def supportsMeshShaders(self):
        """Check if the device supports mesh shaders"""
        return self._context.supportsMeshShaders()
        
    def beginCapture(self):
        """Begin Metal frame capture for debugging"""
        self._context.beginCapture()
        
    def endCapture(self):
        """End Metal frame capture"""
        self._context.endCapture()

# PyMetalScene - Wrapper for MetalScene
cdef class PyMetalScene:
    cdef MetalScene* _scene
    cdef PyMetalContext _context
    
    def __cinit__(self, PyMetalContext context):
        self._context = context
        self._scene = new MetalScene(context._context)
        if self._scene == NULL:
            raise MemoryError("Failed to allocate MetalScene")
    
    def __dealloc__(self):
        if self._scene != NULL:
            del self._scene
            self._scene = NULL
    
    def initialize(self):
        """Initialize the Metal scene"""
        if not self._scene.initialize():
            raise MetalError("Failed to initialize Metal scene")
        return True
        
    def camera(self):
        """Get the camera for the scene"""
        cdef MetalCamera* camera = self._scene.camera()
        if camera == NULL:
            return None
            
        cdef PyMetalCamera pyCamera = PyMetalCamera.__new__(PyMetalCamera)
        pyCamera._camera = camera
        pyCamera._ownsCamera = False
        return pyCamera
        
    def setBackgroundColor(self, float r, float g, float b, float a=1.0):
        """Set the background color of the scene"""
        self._scene.setBackgroundColor(r, g, b, a)
        
    def setAmbientColor(self, float r, float g, float b):
        """Set the ambient light color"""
        self._scene.setAmbientColor(r, g, b)
        
    def setAmbientIntensity(self, float intensity):
        """Set the ambient light intensity"""
        self._scene.setAmbientIntensity(intensity)

# PyMetalCamera - Wrapper for MetalCamera
cdef class PyMetalCamera:
    cdef MetalCamera* _camera
    cdef bool _ownsCamera
    
    def __cinit__(self):
        self._camera = new MetalCamera()
        if self._camera == NULL:
            raise MemoryError("Failed to allocate MetalCamera")
        self._ownsCamera = True
    
    def __dealloc__(self):
        if self._ownsCamera and self._camera != NULL:
            del self._camera
            self._camera = NULL
    
    def setPosition(self, float x, float y, float z):
        """Set the camera position"""
        self._camera.setPosition(x, y, z)
        
    def setTarget(self, float x, float y, float z):
        """Set the camera target (look-at point)"""
        self._camera.setTarget(x, y, z)
        
    def setUp(self, float x, float y, float z):
        """Set the camera up vector"""
        self._camera.setUp(x, y, z)
        
    def setFov(self, float fov):
        """Set the camera field of view (in degrees)"""
        self._camera.setFov(fov)
        
    def setAspectRatio(self, float aspectRatio):
        """Set the camera aspect ratio (width/height)"""
        self._camera.setAspectRatio(aspectRatio)
        
    def setNearPlane(self, float nearPlane):
        """Set the camera near clip plane distance"""
        self._camera.setNearPlane(nearPlane)
        
    def setFarPlane(self, float farPlane):
        """Set the camera far clip plane distance"""
        self._camera.setFarPlane(farPlane)

# PyMetalRenderer - Wrapper for MetalRenderer
cdef class PyMetalRenderer:
    cdef MetalRenderer* _renderer
    cdef PyMetalContext _context
    
    def __cinit__(self, PyMetalContext context):
        self._context = context
        self._renderer = new MetalRenderer(context._context)
        if self._renderer == NULL:
            raise MemoryError("Failed to allocate MetalRenderer")
    
    def __dealloc__(self):
        if self._renderer != NULL:
            del self._renderer
            self._renderer = NULL
    
    def initialize(self):
        """Initialize the Metal renderer"""
        if not self._renderer.initialize():
            raise MetalError("Failed to initialize Metal renderer")
        return True
        
    def setScene(self, PyMetalScene scene):
        """Set the scene to render"""
        self._renderer.setScene(scene._scene)
        
    def beginFrame(self):
        """Begin a new frame for rendering"""
        self._renderer.beginFrame()
        
    def endFrame(self):
        """End the current frame rendering"""
        self._renderer.endFrame()
        
    def setMultiGPUMode(self, bool enabled, int strategy=0):
        """Enable or disable multi-GPU rendering"""
        self._renderer.setMultiGPUMode(enabled, strategy)

# PyMetalMultiGPU - Wrapper for MetalMultiGPU
cdef class PyMetalMultiGPU:
    cdef MetalMultiGPU* _multiGpu
    
    def __cinit__(self):
        self._multiGpu = new MetalMultiGPU()
        if self._multiGpu == NULL:
            raise MemoryError("Failed to allocate MetalMultiGPU")
    
    def __dealloc__(self):
        if self._multiGpu != NULL:
            del self._multiGpu
            self._multiGpu = NULL
    
    def initialize(self, PyMetalContext context):
        """Initialize multi-GPU support"""
        if not self._multiGpu.initialize(context._context):
            raise MetalError("Failed to initialize multi-GPU support")
        return True
        
    def getDeviceInfo(self):
        """Get information about available Metal GPU devices"""
        cdef vector[GPUDeviceInfo] info = self._multiGpu.getDeviceInfo()
        result = []
        
        for i in range(info.size()):
            device = {
                "name": info[i].name.decode('utf-8'),
                "is_primary": info[i].isPrimary,
                "is_active": info[i].isActive,
                "unified_memory": info[i].unifiedMemory,
                "memory_size": info[i].memorySize
            }
            result.append(device)
            
        return result
        
    def enable(self, bool enabled, int strategy=0):
        """Enable or disable multi-GPU rendering"""
        cdef MultiGPUStrategy strat
        
        if strategy == 0:
            strat = SplitFrame
        elif strategy == 1:
            strat = TaskBased
        elif strategy == 2:
            strat = Alternating
        elif strategy == 3:
            strat = ComputeOffload
        else:
            strat = SplitFrame
            
        return self._multiGpu.enable(enabled, strat)
        
    def isEnabled(self):
        """Check if multi-GPU rendering is enabled"""
        return self._multiGpu.isEnabled()
        
    def getStrategy(self):
        """Get the current multi-GPU strategy"""
        cdef MultiGPUStrategy strat = self._multiGpu.getStrategy()
        
        if strat == SplitFrame:
            return 0
        elif strat == TaskBased:
            return 1
        elif strat == Alternating:
            return 2
        elif strat == ComputeOffload:
            return 3
        else:
            return 0

# PyMetalArgBuffer - Wrapper for MetalArgBuffer
cdef class PyMetalArgBuffer:
    cdef shared_ptr[MetalArgBuffer] _argBuffer
    
    def name(self):
        """Get the name of the argument buffer"""
        if not self._argBuffer:
            return ""
        return self._argBuffer.get().name().decode('utf-8')

# Dummy view class for now - in a real implementation, this would wrap the MTKView
cdef class PyMetalView:
    cdef PyMetalContext _context
    cdef PyMetalRenderer _renderer
    cdef PyMetalScene _scene
    
    def __cinit__(self):
        self._context = PyMetalContext()
        self._renderer = None
        self._scene = None
    
    def initialize(self, long window_id, int width, int height):
        """Initialize the Metal view"""
        # Initialize the context
        if not self._context.initialize():
            return False
            
        # Create scene
        self._scene = PyMetalScene(self._context)
        if not self._scene.initialize():
            return False
            
        # Create renderer
        self._renderer = PyMetalRenderer(self._context)
        if not self._renderer.initialize():
            return False
            
        # Set scene in renderer
        self._renderer.setScene(self._scene)
        
        # In a real implementation, we would create an MTKView
        # and attach it to the window_id
        # For now, just return success
        return True
        
    def context(self):
        """Get the Metal context"""
        return self._context
        
    def renderer(self):
        """Get the Metal renderer"""
        return self._renderer
        
    def scene(self):
        """Get the Metal scene"""
        return self._scene
        
    def resize(self, int width, int height):
        """Resize the Metal view"""
        # In a real implementation, we would resize the MTKView
        # For now, just update the camera aspect ratio
        if self._scene and self._scene.camera():
            aspect = float(width) / float(height)
            self._scene.camera().setAspectRatio(aspect)
            
    def render(self):
        """Render the current frame"""
        if self._renderer:
            self._renderer.beginFrame()
            # In a real implementation, the actual rendering would happen here
            # through the MTKView delegate
            self._renderer.endFrame()
            
    def beginCapture(self):
        """Begin Metal frame capture for debugging"""
        if self._context:
            self._context.beginCapture()
            
    def endCapture(self):
        """End Metal frame capture"""
        if self._context:
            self._context.endCapture()

# Module-level functions to check Metal availability
def is_metal_available():
    """Check if Metal is available on the current system"""
    import platform
    if platform.system() != "Darwin":
        return False
        
    # Check macOS version - Metal requires 10.14+ for all features we need
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        major, minor = map(int, mac_ver.split('.')[:2])
        if (major < 10) or (major == 10 and minor < 14):
            return False
            
    return True
