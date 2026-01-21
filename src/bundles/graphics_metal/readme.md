# ChimeraX-GraphicsMetal

Metal-accelerated multi-GPU graphics for UCSF ChimeraX.

## Overview

This bundle provides a high-performance Metal-based graphics renderer for ChimeraX, optimized for macOS systems with Apple Silicon or Intel processors. It leverages Apple's Metal graphics API to deliver improved performance and reduced CPU overhead compared to the default OpenGL renderer.

Key features:
- **Multi-GPU Acceleration**: Automatically distributes rendering workloads across multiple GPUs when available
- **Optimized Memory Usage**: Uses Metal's advanced memory management techniques for better performance
- **Argument Buffers**: Takes advantage of Metal's argument buffers for efficient resource binding
- **Ray Tracing Support**: Optional ray-traced shadows and ambient occlusion on supported hardware
- **Mesh Shaders**: Utilizes Metal mesh shaders for more efficient geometry processing

## Requirements

- macOS 10.14 (Mojave) or later
- ChimeraX 1.5 or later
- A Metal-compatible GPU

## Installation

From within ChimeraX:

1. Open ChimeraX
2. Run the following command:
   ```
   toolshed install ChimeraX-GraphicsMetal
   ```

Alternatively, you can download the wheel file from the releases page and install it using:
```
chimerax --nogui --exit --cmd "toolshed install /path/to/ChimeraX-GraphicsMetal-0.1-py3-none-any.whl"
```

## Usage

### Enabling Metal Rendering

Metal rendering can be enabled with:
```
graphics metal
```

To switch back to OpenGL:
```
graphics opengl
```

### Multi-GPU Acceleration

If your system has multiple GPUs, you can enable multi-GPU acceleration:
```
graphics multigpu true
```

You can also choose a specific strategy for multi-GPU rendering:
```
set metal multiGPUStrategy split-frame
```

Available strategies:
- `split-frame` - Each GPU renders a different portion of the screen
- `task-based` - Different rendering tasks are distributed across GPUs
- `alternating` - Frames are alternated between GPUs
- `compute-offload` - Main GPU handles rendering, other GPUs handle compute tasks

### Preferences

The Metal graphics settings can be configured through the preferences menu:

1. Open ChimeraX
2. Go to `Tools` → `General` → `Metal Graphics`

Or you can use the command interface:
```
set metal useMetal true
set metal autoDetect true
set metal multiGPU true
set metal rayTracing false
```

## Building from Source

### Prerequisites

- macOS 10.14+
- Xcode 12.0+
- Python 3.9+
- Cython 0.29.24+
- NumPy

### Build Steps

1. Clone the repository:
   ```
   git clone https://github.com/alphataubio/ChimeraX-GraphicsMetal.git
   cd ChimeraX-GraphicsMetal
   ```

2. Build the bundle:
   ```
   make build
   ```

3. Install in development mode:
   ```
   make develop
   ```

## Architecture

The bundle implements a Metal-based renderer that integrates with ChimeraX's graphics system:

- `metal_graphics.py` - Python interface to the Metal renderer
- `metal_context.cpp` - Core Metal device and context management
- `metal_renderer.cpp` - Main rendering pipeline implementation
- `metal_scene.cpp` - Scene management for the Metal renderer
- `metal_argbuffer_manager.cpp` - Argument buffer management for efficient resource binding
- `metal_heap_manager.cpp` - Memory management with Metal heaps
- `metal_event_manager.cpp` - Synchronization for multi-GPU rendering
- `metal_multi_gpu.cpp` - Multi-GPU coordination and management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed for non-commercial use only. See the LICENSE file for details.

## Acknowledgments

- UCSF ChimeraX team for their excellent molecular visualization platform
- Apple for the Metal graphics API and development tools
