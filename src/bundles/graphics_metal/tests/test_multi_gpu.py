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
    from chimerax.graphics_metal._metal import (
        PyMetalContext, 
        PyMetalMultiGPU, 
        MetalError
    )
except ImportError:
    print("Metal module not available, skipping tests")
    sys.exit(0)

class TestMultiGPU(unittest.TestCase):
    """Test Metal multi-GPU functionality"""
    
    def setUp(self):
        """Set up test case - create context and multi-GPU manager"""
        self.context = PyMetalContext()
        self.context.initialize()
        
        self.multi_gpu = PyMetalMultiGPU()
        self.initialized = self.multi_gpu.initialize(self.context)
    
    def tearDown(self):
        """Clean up test case"""
        # Objects will be automatically released in __dealloc__
        self.multi_gpu = None
        self.context = None
    
    def test_initialization(self):
        """Test that multi-GPU manager initializes properly"""
        self.assertTrue(self.initialized, "Multi-GPU manager initialization failed")
    
    def test_get_device_info(self):
        """Test that device info is available"""
        # Skip if initialization failed
        if not self.initialized:
            self.skipTest("Multi-GPU manager not initialized")
        
        # Get device info
        devices = self.multi_gpu.getDeviceInfo()
        
        # Verify device info
        self.assertIsNotNone(devices, "Device info is None")
        self.assertIsInstance(devices, list, "Device info is not a list")
        
        # There should be at least one device
        self.assertGreaterEqual(len(devices), 1, "No devices found")
        
        # Check if we have multiple GPUs
        if len(devices) > 1:
            print(f"Found {len(devices)} GPUs:")
            for device in devices:
                print(f"  {device['name']} ({'Primary' if device['is_primary'] else 'Secondary'})")
                # Verify device info keys
                self.assertIn('name', device, "Device info missing 'name'")
                self.assertIn('is_primary', device, "Device info missing 'is_primary'")
                self.assertIn('is_active', device, "Device info missing 'is_active'")
                self.assertIn('unified_memory', device, "Device info missing 'unified_memory'")
                self.assertIn('memory_size', device, "Device info missing 'memory_size'")
        else:
            print("Only one GPU found, multi-GPU tests will be limited")
    
    def test_enable_disable(self):
        """Test enabling and disabling multi-GPU"""
        # Skip if initialization failed
        if not self.initialized:
            self.skipTest("Multi-GPU manager not initialized")
        
        # Get device info to see if we have multiple GPUs
        devices = self.multi_gpu.getDeviceInfo()
        has_multiple_gpus = len(devices) > 1
        
        # Enable multi-GPU
        enabled = self.multi_gpu.enable(True)
        if has_multiple_gpus:
            self.assertTrue(enabled, "Failed to enable multi-GPU with multiple GPUs")
            self.assertTrue(self.multi_gpu.isEnabled(), "Multi-GPU not enabled after enable(True)")
        else:
            # With only one GPU, enable() may return False
            print("Note: enable() may return False with only one GPU")
        
        # Disable multi-GPU
        self.multi_gpu.enable(False)
        self.assertFalse(self.multi_gpu.isEnabled(), "Multi-GPU still enabled after enable(False)")
    
    def test_strategies(self):
        """Test setting different multi-GPU strategies"""
        # Skip if initialization failed
        if not self.initialized:
            self.skipTest("Multi-GPU manager not initialized")
        
        # Get device info to see if we have multiple GPUs
        devices = self.multi_gpu.getDeviceInfo()
        has_multiple_gpus = len(devices) > 1
        
        if not has_multiple_gpus:
            self.skipTest("Multiple GPUs required for strategy tests")
        
        # Test each strategy
        for strategy in range(4):
            enabled = self.multi_gpu.enable(True, strategy)
            self.assertTrue(enabled, f"Failed to enable multi-GPU with strategy {strategy}")
            self.assertTrue(self.multi_gpu.isEnabled(), f"Multi-GPU not enabled after enable(True, {strategy})")
            
            # Verify strategy was set
            current_strategy = self.multi_gpu.getStrategy()
            self.assertEqual(current_strategy, strategy, f"Strategy not set correctly, expected {strategy}, got {current_strategy}")
            
            # Disable multi-GPU after testing each strategy
            self.multi_gpu.enable(False)
            self.assertFalse(self.multi_gpu.isEnabled(), "Multi-GPU still enabled after enable(False)")

if __name__ == '__main__':
    unittest.main()
