# Makefile for ChimeraX-GraphicsMetal bundle

PYTHON = python3
CHIMERAX = chimerax
BUNDLE_NAME = ChimeraX-GraphicsMetal
BUNDLE_VERSION = 0.1

.PHONY: clean build install test wheel debug develop lint

# Default target
all: build

# Build the bundle
build:
	$(PYTHON) -m pip install -e .

# Create a wheel package
wheel:
	$(PYTHON) -m build --wheel

# Install the bundle in the current ChimeraX environment
install:
	$(PYTHON) -m pip install .

# Install in development mode
develop:
	$(PYTHON) -m pip install -e .

# Install the bundle in ChimeraX
install-chimerax: wheel
	$(CHIMERAX) --nogui --exit --cmd "toolshed install dist/$(BUNDLE_NAME)-$(BUNDLE_VERSION)-*.whl"

# Run tests
test:
	$(PYTHON) -m pytest tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/chimerax/graphics_metal/__pycache__/
	rm -rf tests/__pycache__/
	rm -f src/cython/*.c
	rm -f src/cython/*.cpp
	rm -f src/cython/*.html
	rm -f src/chimerax/graphics_metal/_metal*.so

# Build with debug info
debug:
	CFLAGS="-O0 -g" $(PYTHON) -m pip install -e .

# Compile Metal shaders
shaders:
	@echo "Compiling Metal shaders..."
	@mkdir -p build/metallib
	for file in metal/shaders/*.metal; do \
		xcrun -sdk macosx metal -c $$file -o build/metallib/$$(basename $$file .metal).air; \
	done
	xcrun -sdk macosx metallib build/metallib/*.air -o src/chimerax/graphics_metal/metal_shaders/default.metallib

# Run linter
lint:
	flake8 src tests
