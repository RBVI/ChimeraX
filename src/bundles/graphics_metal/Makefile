BUILD_AND_INSTALL = 1
include ../Makefile.bundle

# Compile Metal shaders to .metallib alongside the bundle build.
# xcrun is only available on macOS; on other platforms this is a no-op.
METALLIB_DIR = src/metal_shaders
METALLIB     = $(METALLIB_DIR)/default.metallib
METAL_SRCS   = $(wildcard metal/shaders/*.metal)

# Hook shader compilation in before the standard install target.
app-install: shaders

shaders: $(METALLIB)

$(METALLIB): $(METAL_SRCS)
	@mkdir -p build/metallib $(METALLIB_DIR)
	@for f in $(METAL_SRCS); do \
		xcrun -sdk macosx metal -c -std=metal3.0 $$f \
		    -o build/metallib/$$(basename $$f .metal).air 2>/dev/null || \
		xcrun -sdk macosx metal -c $$f \
		    -o build/metallib/$$(basename $$f .metal).air; \
	done
	xcrun -sdk macosx metallib build/metallib/*.air -o $(METALLIB)

clean:: clean-shaders

clean-shaders:
	rm -rf build/metallib $(METALLIB)
