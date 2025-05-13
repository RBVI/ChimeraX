"""
Preferences for ChimeraX Metal graphics acceleration
"""

from chimerax.core.settings import Settings
from chimerax.core.commands import BoolArg, EnumOf
from chimerax.core.commands import register as register_command

# Enum argument for multi-GPU strategy
MultiGPUStrategyArg = EnumOf(["split-frame", "task-based", "alternating", "compute-offload"])

class _MetalGraphicsSettings(Settings):
    EXPLICIT_SAVE = {
        'use_metal': True,
        'auto_detect': True, 
        'multi_gpu_enabled': False,
        'multi_gpu_strategy': 'split-frame',
        'enable_mesh_shaders': True,
        'enable_ray_tracing': False,
        'enable_argument_buffers': True,
        'prefer_device_local_memory': True,
    }
    
    use_metal = Settings.BoolSetting(
        "Use Metal for graphics rendering",
        True,
        "Whether to use Metal for graphics rendering on macOS",
    )
    
    auto_detect = Settings.BoolSetting(
        "Auto-detect Metal support",
        True,
        "Automatically detect and enable Metal if supported",
    )
    
    multi_gpu_enabled = Settings.BoolSetting(
        "Enable Multi-GPU acceleration",
        False, 
        "Use multiple GPUs for rendering if available",
    )
    
    multi_gpu_strategy = Settings.EnumSetting(
        "Multi-GPU rendering strategy",
        "split-frame",
        ["split-frame", "task-based", "alternating", "compute-offload"],
        "Strategy to use for multi-GPU rendering",
    )
    
    enable_mesh_shaders = Settings.BoolSetting(
        "Enable mesh shaders",
        True,
        "Use mesh shaders for more efficient geometry rendering",
    )
    
    enable_ray_tracing = Settings.BoolSetting(
        "Enable ray tracing",
        False,
        "Use ray tracing for higher quality lighting and shadows",
    )
    
    enable_argument_buffers = Settings.BoolSetting(
        "Enable argument buffers",
        True,
        "Use argument buffers for more efficient resource binding",
    )
    
    prefer_device_local_memory = Settings.BoolSetting(
        "Prefer device local memory",
        True,
        "Prefer device local memory over shared memory for better performance",
    )

# Singleton instance of settings
settings = None

def register_settings(session):
    """Register Metal graphics settings with ChimeraX"""
    global settings
    if settings is None:
        settings = _MetalGraphicsSettings(session, "metal_graphics")
    return settings

def register_metal_preferences(session):
    """Register Metal preferences UI with ChimeraX"""
    from chimerax.ui.gui import MainToolWindow
    from chimerax.core.commands import run
    
    # Get settings
    prefs = register_settings(session)
    
    # Register preference commands
    register_command(
        session,
        "set metal",
        set_metal_settings,
        help="Set Metal rendering preferences"
    )
    
    # Function to register preferences UI
    def _register_ui():
        from Qt.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QCheckBox, QComboBox, QLabel
        
        class _MetalPrefsUI(MainToolWindow):
            SESSION_ENDURING = True
            SESSION_SAVE = True
            help = "help:user/tools/metalgraphics.html"
            
            def __init__(self, session, tool_name):
                super().__init__(session, tool_name)
                
                # Create main widget
                self.ui_area = QWidget()
                layout = QVBoxLayout()
                self.ui_area.setLayout(layout)
                self.setCentralWidget(self.ui_area)
                
                # Create form layout for settings
                form = QFormLayout()
                layout.addLayout(form)
                
                # Use Metal checkbox
                self.use_metal_cb = QCheckBox()
                self.use_metal_cb.setChecked(prefs.use_metal)
                self.use_metal_cb.clicked.connect(self._use_metal_changed)
                form.addRow("Use Metal rendering:", self.use_metal_cb)
                
                # Auto-detect checkbox
                self.auto_detect_cb = QCheckBox()
                self.auto_detect_cb.setChecked(prefs.auto_detect)
                self.auto_detect_cb.clicked.connect(self._auto_detect_changed)
                form.addRow("Auto-detect Metal support:", self.auto_detect_cb)
                
                # Multi-GPU checkbox
                self.multi_gpu_cb = QCheckBox()
                self.multi_gpu_cb.setChecked(prefs.multi_gpu_enabled)
                self.multi_gpu_cb.clicked.connect(self._multi_gpu_changed)
                form.addRow("Enable Multi-GPU acceleration:", self.multi_gpu_cb)
                
                # Multi-GPU strategy
                self.strategy_combo = QComboBox()
                self.strategy_combo.addItems([
                    "Split Frame", 
                    "Task Based", 
                    "Alternating Frames", 
                    "Compute Offload"
                ])
                strategy_map = {
                    "split-frame": 0,
                    "task-based": 1,
                    "alternating": 2,
                    "compute-offload": 3
                }
                self.strategy_combo.setCurrentIndex(strategy_map.get(prefs.multi_gpu_strategy, 0))
                self.strategy_combo.currentIndexChanged.connect(self._strategy_changed)
                form.addRow("Multi-GPU strategy:", self.strategy_combo)
                
                # Mesh shaders checkbox
                self.mesh_shaders_cb = QCheckBox()
                self.mesh_shaders_cb.setChecked(prefs.enable_mesh_shaders)
                self.mesh_shaders_cb.clicked.connect(self._mesh_shaders_changed)
                form.addRow("Enable mesh shaders:", self.mesh_shaders_cb)
                
                # Ray tracing checkbox
                self.ray_tracing_cb = QCheckBox()
                self.ray_tracing_cb.setChecked(prefs.enable_ray_tracing)
                self.ray_tracing_cb.clicked.connect(self._ray_tracing_changed)
                form.addRow("Enable ray tracing:", self.ray_tracing_cb)
                
                # Argument buffers checkbox
                self.arg_buffers_cb = QCheckBox()
                self.arg_buffers_cb.setChecked(prefs.enable_argument_buffers)
                self.arg_buffers_cb.clicked.connect(self._arg_buffers_changed)
                form.addRow("Enable argument buffers:", self.arg_buffers_cb)
                
                # Device local memory checkbox
                self.device_local_cb = QCheckBox()
                self.device_local_cb.setChecked(prefs.prefer_device_local_memory)
                self.device_local_cb.clicked.connect(self._device_local_changed)
                form.addRow("Prefer device local memory:", self.device_local_cb)
                
                # Add hardware info
                self._add_hardware_info(layout)
                
                self.manage(None)
            
            def _add_hardware_info(self, layout):
                """Add hardware info section"""
                # Try to get Metal hardware info
                try:
                    from . import _metal
                    if not _metal.is_metal_available():
                        layout.addWidget(QLabel("Metal is not available on this system"))
                        return
                        
                    context = _metal.PyMetalContext()
                    if not context.initialize():
                        layout.addWidget(QLabel("Failed to initialize Metal context"))
                        return
                        
                    # Add Metal device info
                    layout.addWidget(QLabel(f"Metal Device: {context.deviceName()}"))
                    layout.addWidget(QLabel(f"Vendor: {context.deviceVendor()}"))
                    layout.addWidget(QLabel(f"Unified Memory: {'Yes' if context.supportsUnifiedMemory() else 'No'}"))
                    layout.addWidget(QLabel(f"Ray Tracing: {'Yes' if context.supportsRayTracing() else 'No'}"))
                    layout.addWidget(QLabel(f"Mesh Shaders: {'Yes' if context.supportsMeshShaders() else 'No'}"))
                    
                    # Get multi-GPU info
                    multi_gpu = _metal.PyMetalMultiGPU()
                    if multi_gpu.initialize(context):
                        devices = multi_gpu.getDeviceInfo()
                        if len(devices) > 1:
                            layout.addWidget(QLabel(f"Multiple GPUs: {len(devices)} devices"))
                            for device in devices:
                                layout.addWidget(QLabel(f"  - {device['name']} ({'Primary' if device['is_primary'] else 'Secondary'})"))
                        else:
                            layout.addWidget(QLabel("Multiple GPUs: No"))
                except Exception as e:
                    layout.addWidget(QLabel(f"Failed to get Metal info: {str(e)}"))
            
            def _use_metal_changed(self, checked):
                prefs.use_metal = checked
                run(self.session, f"set metal useMetal {checked}")
                
            def _auto_detect_changed(self, checked):
                prefs.auto_detect = checked
                run(self.session, f"set metal autoDetect {checked}")
                
            def _multi_gpu_changed(self, checked):
                prefs.multi_gpu_enabled = checked
                run(self.session, f"set metal multiGPU {checked}")
                
            def _strategy_changed(self, index):
                strategies = ["split-frame", "task-based", "alternating", "compute-offload"]
                if 0 <= index < len(strategies):
                    prefs.multi_gpu_strategy = strategies[index]
                    run(self.session, f"set metal multiGPUStrategy {strategies[index]}")
                
            def _mesh_shaders_changed(self, checked):
                prefs.enable_mesh_shaders = checked
                run(self.session, f"set metal meshShaders {checked}")
                
            def _ray_tracing_changed(self, checked):
                prefs.enable_ray_tracing = checked
                run(self.session, f"set metal rayTracing {checked}")
                
            def _arg_buffers_changed(self, checked):
                prefs.enable_argument_buffers = checked
                run(self.session, f"set metal argumentBuffers {checked}")
                
            def _device_local_changed(self, checked):
                prefs.prefer_device_local_memory = checked
                run(self.session, f"set metal deviceLocalMemory {checked}")
        
        # Register the preferences tool
        session.tools.register("Metal Graphics", _MetalPrefsUI, None, None, None)
    
    # Register UI if GUI is available
    if hasattr(session, 'ui') and session.ui.is_gui:
        _register_ui()

def set_metal_settings(session, useMetal=None, autoDetect=None, multiGPU=None, 
                      multiGPUStrategy=None, meshShaders=None, rayTracing=None, 
                      argumentBuffers=None, deviceLocalMemory=None):
    """Set Metal rendering preferences"""
    prefs = register_settings(session)
    
    # Update settings that were specified
    if useMetal is not None:
        prefs.use_metal = useMetal
        
    if autoDetect is not None:
        prefs.auto_detect = autoDetect
        
    if multiGPU is not None:
        prefs.multi_gpu_enabled = multiGPU
        
    if multiGPUStrategy is not None:
        prefs.multi_gpu_strategy = multiGPUStrategy
        
    if meshShaders is not None:
        prefs.enable_mesh_shaders = meshShaders
        
    if rayTracing is not None:
        prefs.enable_ray_tracing = rayTracing
        
    if argumentBuffers is not None:
        prefs.enable_argument_buffers = argumentBuffers
        
    if deviceLocalMemory is not None:
        prefs.prefer_device_local_memory = deviceLocalMemory
    
    # Apply settings if graphics is active
    from chimerax.graphics_metal.metal_graphics import MetalGraphics
    if isinstance(session.main_view.graphics, MetalGraphics):
        graphics = session.main_view.graphics
        
        # Apply multi-GPU settings
        if multiGPU is not None and prefs.multi_gpu_enabled:
            strategy_map = {
                "split-frame": 0,
                "task-based": 1,
                "alternating": 2,
                "compute-offload": 3
            }
            strategy = strategy_map.get(prefs.multi_gpu_strategy, 0)
            graphics.enable_multi_gpu(True, strategy)
        elif multiGPU is not None and not prefs.multi_gpu_enabled:
            graphics.enable_multi_gpu(False)
    
    # Report current settings
    session.logger.info(f"Metal Graphics Settings:")
    session.logger.info(f"  Use Metal: {prefs.use_metal}")
    session.logger.info(f"  Auto-detect: {prefs.auto_detect}")
    session.logger.info(f"  Multi-GPU enabled: {prefs.multi_gpu_enabled}")
    session.logger.info(f"  Multi-GPU strategy: {prefs.multi_gpu_strategy}")
    session.logger.info(f"  Mesh shaders: {prefs.enable_mesh_shaders}")
    session.logger.info(f"  Ray tracing: {prefs.enable_ray_tracing}")
    session.logger.info(f"  Argument buffers: {prefs.enable_argument_buffers}")
    session.logger.info(f"  Prefer device local memory: {prefs.prefer_device_local_memory}")

# Register commands
set_metal_settings.register_arguments = [
    ('useMetal', BoolArg, 'Whether to use Metal for rendering'),
    ('autoDetect', BoolArg, 'Automatically detect Metal support'),
    ('multiGPU', BoolArg, 'Enable multi-GPU acceleration'),
    ('multiGPUStrategy', MultiGPUStrategyArg, 'Multi-GPU rendering strategy'),
    ('meshShaders', BoolArg, 'Enable mesh shaders'),
    ('rayTracing', BoolArg, 'Enable ray tracing'),
    ('argumentBuffers', BoolArg, 'Enable argument buffers'),
    ('deviceLocalMemory', BoolArg, 'Prefer device local memory')
]
