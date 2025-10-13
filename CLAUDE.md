# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ChimeraX is a molecular visualization application built with a plugin architecture called "bundles". The codebase is ~80% Python 3.11, ~20% C++, using Qt for the GUI.

## Build Commands

### Main Build
```bash
make install          # Full build including 50+ third-party dependencies
make build-minimal    # Core components only
make uv-build         # Modern uv-based build (faster)
```

### Testing
```bash
make test             # Run all tests
make src.test         # Run source tests only
pytest                # Run pytest directly
```

### Development
```bash
make clean            # Clean build artifacts
make distclean        # Full clean including dependencies
```

## Bundle Development

ChimeraX functionality is organized into "bundles" (plugins). Each bundle is a Python package with specific metadata.

### Bundle Structure
```
bundle_name/
├── pyproject.toml    # Modern config (preferred)
├── bundle_info.xml   # Legacy config
├── src/              # Bundle source code
│   ├── __init__.py   # BundleAPI implementation
│   ├── cmd.py        # Command implementations
│   └── tool.py       # GUI tools
├── tests/            # Bundle tests
└── Makefile          # Build configuration
```

### Bundle Development Commands
```bash
# From within a bundle directory:
make install          # Build and install bundle
make wheel            # Build Python wheel
make test             # Run bundle tests
make install-editable # Install a bundle in editable mode for development
make uv-install       # Fast install with uv
make uv-install-editable  # Editable install for development

# Using ChimeraX devel commands:
devel install PATH    # Build and install bundle
devel build PATH      # Build wheel only
devel clean PATH      # Clean build files
```

### Bundle Configuration

Modern bundles use `pyproject.toml` with ChimeraX-specific sections:

```toml
[tool.chimerax]
min-session-version = 1
max-session-version = 1
categories = ["General"]

[tool.chimerax.command.mycommand]
category = "General"
description = "My custom command"

[tool.chimerax.tool."My Tool"]
category = "General"
description = "My custom tool"
```

## Architecture

### Core Components
- **src/bundles/core/**: Fundamental APIs and infrastructure
- **src/bundles/ui/**: User interface framework
- **src/bundles/atomic/**: Molecular structure representation
- **src/bundles/graphics/**: 3D rendering and visualization

### Bundle System
- Bundles declare dependencies in their configuration
- Runtime lazy loading for performance
- Each bundle implements a `BundleAPI` class as the interface to ChimeraX core
- Commands, tools, file formats, and other features are registered through bundle APIs

### Build System
- GNU Make with platform-specific configurations (Windows, macOS, Linux)
- Complex dependency management through `prereqs/` system
- Both traditional make and modern uv-based workflows supported

## Key Directories

- **src/bundles/**: All bundle implementations (~100+ bundles)
- **src/apps/**: Application launchers and executables
- **prereqs/**: Third-party dependencies and custom Python build
- **mk/**: Build system configuration files
- **docs/**: Developer and user documentation

## Development Notes

- Bundle APIs must be implemented in `src/__init__.py`
- Use `APP_PYTHON_EXE` variable for the ChimeraX Python executable
- Tests use pytest with special ChimeraX session fixtures
- Modern bundles should prefer `pyproject.toml` over `bundle_info.xml`
- The `toolshed` command manages bundle installation/removal in built ChimeraX
