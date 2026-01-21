#!/usr/bin/env python

import unittest
import sys
import platform
import os

# Skip tests if not on macOS
if platform.system() != "Darwin":
    print("Skipping Metal tests on non-macOS platform")
    sys.exit(0)

try:
    from chimerax.graphics_metal._metal import PyMetalContext, MetalError
except ImportError:
    print("Metal module not available, skipping tests")
    sys.exit(0)

class TestMetalContext(unittest.TestCase):
    """Test Metal context creation and functionality"""
    
    def setUp(self):
        """Set up test case - create context"""
        self.context = PyMetalContext()
    
    def tearDown(self):
        """Clean up test case"""
        # Context will be automatically released in __dealloc__
        self.context = None
    
    def test_initialization(self):
        """Test that context initializes properly"""
        # Initialize context
        initialized = self.context.initialize()
        self.assertTrue(initialized, "Metal context initialization failed")
        self.assertTrue(self.context.isInitialized(), "Context reports not initialized after successful initialization")
    
    def test_device_info(self):
        """Test that device info is available"""
        # Initialize context
        self.context.initialize()
        
        # Get device info
        device_name = self.context.deviceName()
        device_vendor = self.context.deviceVendor()
        
        self.assertIsNotNone(device_name, "Device name is None")
        self.assertNotEqual(device_name, "", "Device name is empty")
        self.assertIsNotNone(device_vendor, "Device vendor is None")
        self.assertNotEqual(device_vendor, "", "Device vendor is empty")
        
        # Log device info for debugging
        print(f"Metal Device: {device_name}")
        print(f"Metal Vendor: {device_vendor}")
    
    def test_device_capabilities(self):
        """Test device capability reporting"""
        # Initialize context
        self.context.initialize()
        
        # Get capabilities
        unified_memory = self.context.supportsUnifiedMemory()
        ray_tracing = self.context.supportsRayTracing()
        mesh_shaders = self.context.supportsMeshShaders()
        
        # These could be True or False depending on hardware
        # Just verify they return boolean values
        self.assertIsInstance(unified_memory, bool, "Unified memory support is not a boolean")
        self.assertIsInstance(ray_tracing, bool, "Ray tracing support is not a boolean")
        self.assertIsInstance(mesh_shaders, bool, "Mesh shader support is not a boolean")
        
        # Log capabilities for debugging
        print(f"Unified Memory: {unified_memory}")
        print(f"Ray Tracing: {ray_tracing}")
        print(f"Mesh Shaders: {mesh_shaders}")

if __name__ == '__main__':
    unittest.main()
