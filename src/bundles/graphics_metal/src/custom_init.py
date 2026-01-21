"""
Custom initialization for ChimeraX-GraphicsMetal bundle.
This is called when the bundle is loaded through the 'custom-init' entry in pyproject.toml.
"""

def init(session, bundle_info):
    """Initialize the Metal graphics bundle"""
    from . import metal_graphics
    
    # Register preferences
    from . import preferences
    preferences.register_settings(session)
    preferences.register_metal_preferences(session)
    
    # Check if Metal is supported
    if metal_graphics.is_metal_supported():
        session.logger.info("Metal graphics acceleration is available")
        
        # Register commands for Metal
        from chimerax.core.commands import register
        register('graphics metal', switch_to_metal,
                 help="Switch to Metal graphics renderer")
        register('graphics opengl', switch_to_opengl,
                 help="Switch back to OpenGL graphics renderer")
        register('graphics multigpu', toggle_multi_gpu,
                 help="Toggle multi-GPU acceleration for Metal renderer")
        
        # Auto-enable Metal if set in preferences
        prefs = preferences.register_settings(session)
        if prefs.auto_detect and prefs.use_metal:
            try:
                switch_to_metal(session)
                session.logger.info("Auto-enabled Metal graphics renderer")
            except Exception as e:
                session.logger.warning(f"Failed to auto-enable Metal: {str(e)}")
    else:
        session.logger.info("Metal graphics acceleration is not available on this system")

def finish(session, bundle_info):
    """Clean up when bundle is unloaded"""
    from chimerax.graphics import provider_info
    
    # Check if we're using Metal
    from .metal_graphics import MetalGraphics
    if isinstance(session.main_view.graphics, MetalGraphics):
        # Switch back to OpenGL before unloading
        default_provider = provider_info('opengl')
        if default_provider:
            session.main_view.switch_graphics_provider('opengl')
            session.logger.info("Switched back to OpenGL graphics renderer")

def switch_to_metal(session):
    """Command to switch to Metal renderer"""
    from . import metal_graphics
    if not metal_graphics.is_metal_supported():
        from chimerax.core.errors import UserError
        raise UserError("Metal graphics is not supported on this system")
        
    if session.main_view.graphics_changed:
        # Graphics provider already changed, check if it's Metal
        from .metal_graphics import MetalGraphics
        if isinstance(session.main_view.graphics, MetalGraphics):
            session.logger.info("Already using Metal graphics")
            return
            
    # Switch to Metal
    from chimerax.graphics import provider_info
    metal_provider = provider_info('metal')
    if metal_provider:
        session.main_view.switch_graphics_provider('metal')
        session.logger.info("Switched to Metal graphics renderer")
        
        # Apply multi-GPU setting from preferences
        from . import preferences
        prefs = preferences.register_settings(session)
        if prefs.multi_gpu_enabled:
            graphics = session.main_view.graphics
            strategy_map = {
                "split-frame": 0,
                "task-based": 1,
                "alternating": 2,
                "compute-offload": 3
            }
            strategy = strategy_map.get(prefs.multi_gpu_strategy, 0)
            if hasattr(graphics, 'enable_multi_gpu'):
                graphics.enable_multi_gpu(True, strategy)
                session.logger.info(f"Enabled multi-GPU acceleration with strategy: {prefs.multi_gpu_strategy}")
    else:
        from chimerax.core.errors import UserError
        raise UserError("Metal graphics provider not found")
        
def switch_to_opengl(session):
    """Command to switch back to OpenGL renderer"""
    # Switch to default OpenGL
    from .metal_graphics import MetalGraphics
    if isinstance(session.main_view.graphics, MetalGraphics):
        session.main_view.switch_graphics_provider('opengl')
        session.logger.info("Switched to OpenGL graphics renderer")
    else:
        session.logger.info("Already using OpenGL graphics")
        
def toggle_multi_gpu(session, enable=None):
    """Command to toggle multi-GPU acceleration"""
    from .metal_graphics import MetalGraphics
    if not isinstance(session.main_view.graphics, MetalGraphics):
        from chimerax.core.errors import UserError
        raise UserError("Multi-GPU acceleration is only available with Metal graphics")
        
    graphics = session.main_view.graphics
    if enable is None:
        # Toggle current state
        enable = not graphics.is_multi_gpu_enabled()
        
    success = graphics.enable_multi_gpu(enable)
    if success:
        session.logger.info(f"Multi-GPU acceleration {'enabled' if enable else 'disabled'}")
        
        # Update preference
        from . import preferences
        prefs = preferences.register_settings(session)
        prefs.multi_gpu_enabled = enable
    else:
        if enable:
            session.logger.warning("Failed to enable multi-GPU acceleration")
        else:
            session.logger.info("Multi-GPU acceleration disabled")
            
            # Update preference
            from . import preferences
            prefs = preferences.register_settings(session)
            prefs.multi_gpu_enabled = False
