# AGENTS.md

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


## Tips
**Compilation, build, and test commands can be very verbose.** This is a GUI application that is difficult to validate without eyes. Always redirect build and test commands to a log file and search for the output you require. Do not pollute your context with bare runs printing to stdout.

**Avoid overindexing on microoptimizations.** When analyzing the performance of an algorithm and looking for ways to improve runtime, consider alternative algorithms, newer algorithms, parallelization, and avoiding repeated work. 

**End to end testing counts.** While local speed increases are always appreciated, the user-perceptible latency of an action is the most important consideration, from command to output.

**Preserve product quality while optimizing.** It is not acceptable to use a worse solution to improve runtime, e.g. using lower precision calculations. Before making changes, explain the suggestion and the rationale for approval.

**Do not excessively comment code.** Agent-generated comments can form a feedback loop where the agent becomes convinced that the comment it just wrote is the law of the land. Comment sparingly, comment about intent, do not simply restate what the code does.

**Keep changes as small as possible.** Commit early and often. Each commit should be the smallest coherent, buildable change, including its tests and necessary documentation, that a human can review. Keep coupled changes together and record prerequisite commits when integration order matters.

**All commits must build.** Do not leave the source code in a broken state between commits.

**Suggest improvements to local code when given a bug.** But do not autonomously change APIs or contracts.

## Development Notes

### Python linting

- Install the pinned linter with `python -m pip install -r utils/lint-requirements.txt`; this does not require building ChimeraX.
- Run `python -m ruff check --config pyproject.toml PATH...` on changed Python files, including new files. Use the explicit configuration so local runs and CI share the same policy across bundles.
- CI checks whole changed files, so existing findings in those files may surface. Fix relevant issues; do not blanket-suppress findings or mix unrelated cleanup into the change. A manual workflow run checks the full tree to inventory remaining debt.
- The initial rules catch syntax errors, undefined names, invalid control flow, always-true tuple assertions, literal identity comparisons, mutable defaults, and loop-variable closure bugs. Formatting, import sorting, unused imports, and unused variables are intentionally outside this policy.
- Keep suppressions specific to a rule and explain the reason. Do not automatically apply fixes or reformat unrelated code. Add rules only when they encode an agreed decision with actionable, low-noise findings.

### Bundle conventions

- Bundle APIs must be implemented in `src/__init__.py`
- Use `APP_PYTHON_EXE` variable for the ChimeraX Python executable
- Tests use pytest with special ChimeraX session fixtures
- Modern bundles should prefer `pyproject.toml` over `bundle_info.xml`
- The `toolshed` command manages bundle installation/removal in built ChimeraX
- **When modifying code, add tests to cover the changes.** Tests go in `tests/` at the repo root or in bundle-specific `tests/` directories.
- **Do not add yourself to commits**
