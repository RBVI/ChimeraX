#!/usr/bin/env python

"""
Integration tests for ChimeraX Metal graphics

These tests require a running ChimeraX session with the graphics_metal
bundle installed and use the ChimeraX API to test the Metal renderer.
"""

import unittest
import sys
import platform
import os
import time

# Skip tests if not on macOS
if platform.system() != "Darwin":
    print("Skipping Metal tests on non-macOS platform")
    sys.exit(0)

# Test if we're running inside ChimeraX
try:
    import chimerax
    from chimerax.core.commands import run
    from chimerax.graphics_metal import metal_graphics
except ImportError:
    print("Not running in ChimeraX or graphics_metal bundle not installed, skipping tests")
    sys.exit(0)

class TestMetalIntegration(unittest.TestCase):
    """Integration tests for Metal graphics in ChimeraX"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - get ChimeraX session"""
        # This requires running inside ChimeraX
        try:
            from chimerax.core.session import session as chimerax_session
            cls.session = chimerax_session
        except ImportError:
            cls.session = None
    
    def setUp(self):
        """Set up test case - ensure we have a session"""
        if self.session is None:
            self.skipTest("No ChimeraX session available")
    
    def test_metal_availability(self):
        """Test Metal availability detection"""
        # Check if Metal is available
        is_supported = metal_graphics.is_metal_supported()
        self.assertIsInstance(is_supported, bool, "Metal support check did not return a boolean")
        
        # Log availability for debugging
        print(f"Metal support: {'Available' if is_supported else 'Not available'}")
    
    def test_switch_to_metal(self):
        """Test switching to Metal graphics provider"""
        # Check if Metal is supported first
        if not metal_graphics.is_metal_supported():
            self.skipTest("Metal not supported on this system")
        
        # Save current graphics provider
        original_provider = self.session.main_view.graphics_provider_info.name
        
        try:
            # Switch to Metal
            run(self.session, "graphics metal")
            
            # Verify switch was successful
            self.assertIsInstance(
                self.session.main_view.graphics, 
                metal_graphics.MetalGraphics,
                "Failed to switch to Metal graphics provider"
            )
            
            # Get capabilities
            capabilities = self.session.main_view.graphics.get_capabilities()
            self.assertEqual(capabilities["api"], "Metal", "Graphics API is not Metal")
            
            # Log capabilities for debugging
            print("Metal graphics capabilities:")
            for key, value in capabilities.items():
                print(f"  {key}: {value}")
            
            # Test rendering a simple model
            run(self.session, "open 1ubq")
            
            # Give it time to render
            time.sleep(1)
            
            # Change representation
            run(self.session, "cartoon")
            
            # Give it time to render
            time.sleep(1)
            
            # Try different camera view
            run(self.session, "view")
            
            # Give it time to render
            time.sleep(1)
        
        finally:
            # Switch back to original provider
            run(self.session, f"graphics {original_provider}")
    
    def test_multi_gpu_toggling(self):
        """Test enabling and disabling multi-GPU rendering"""
        # Check if Metal is supported first
        if not metal_graphics.is_metal_supported():
            self.skipTest("Metal not supported on this system")
        
        # Save current graphics provider
        original_provider = self.session.main_view.graphics_provider_info.name
        
        try:
            # Switch to Metal
            run(self.session, "graphics metal")
            
            # Verify switch was successful
            self.assertIsInstance(
                self.session.main_view.graphics, 
                metal_graphics.MetalGraphics,
                "Failed to switch to Metal graphics provider"
            )
            
            # Check if multi-GPU is supported
            capabilities = self.session.main_view.graphics.get_capabilities()
            if not capabilities.get("multi_gpu", False):
                self.skipTest("Multi-GPU not supported on this system")
            
            # Enable multi-GPU
            run(self.session, "graphics multigpu true")
            
            # Verify multi-GPU is enabled
            self.assertTrue(
                self.session.main_view.graphics.is_multi_gpu_enabled(),
                "Failed to enable multi-GPU rendering"
            )
            
            # Test rendering with multi-GPU
            run(self.session, "open 1ubq")
            
            # Give it time to render
            time.sleep(1)
            
            # Change representation
            run(self.session, "cartoon")
            
            # Give it time to render
            time.sleep(1)
            
            # Disable multi-GPU
            run(self.session, "graphics multigpu false")
            
            # Verify multi-GPU is disabled
            self.assertFalse(
                self.session.main_view.graphics.is_multi_gpu_enabled(),
                "Failed to disable multi-GPU rendering"
            )
        
        finally:
            # Switch back to original provider
            run(self.session, f"graphics {original_provider}")
    
    def test_preferences(self):
        """Test Metal preferences"""
        # Get preferences
        from chimerax.graphics_metal.preferences import register_settings
        prefs = register_settings(self.session)
        
        # Save original values
        original_use_metal = prefs.use_metal
        original_multi_gpu = prefs.multi_gpu_enabled
        
        try:
            # Change preferences
            prefs.use_metal = not original_use_metal
            prefs.multi_gpu_enabled = not original_multi_gpu
            
            # Verify changes were saved
            self.assertEqual(prefs.use_metal, not original_use_metal, "Failed to change use_metal preference")
            self.assertEqual(prefs.multi_gpu_enabled, not original_multi_gpu, "Failed to change multi_gpu_enabled preference")
            
            # Test command-based preference setting
            run(self.session, f"set metal useMetal {original_use_metal}")
            run(self.session, f"set metal multiGPU {original_multi_gpu}")
            
            # Verify command-based changes were saved
            self.assertEqual(prefs.use_metal, original_use_metal, "Failed to change use_metal preference via command")
            self.assertEqual(prefs.multi_gpu_enabled, original_multi_gpu, "Failed to change multi_gpu_enabled preference via command")
        
        finally:
            # Restore original values
            prefs.use_metal = original_use_metal
            prefs.multi_gpu_enabled = original_multi_gpu

if __name__ == '__main__':
    unittest.main()
