TOP = ../../..
include $(TOP)/mk/config.make

include ../Makefile.bundle

# HoloPlayCore native libraries come from the Looking Glass Core SDK:
#   https://github.com/Looking-Glass/LookingGlassCoreSDK/releases/tag/v0.1.0
# The only attached release asset is a Windows installer, so we pull the
# prebuilt Win64/Linux/macOS libraries out of the source archive for the tag.
# These are the same files (byte for byte) previously mirrored on the RBVI
# prereqs server.
HOLOPLAYCORE_VERSION = 0.1.0
HOLOPLAYCORE_TARBALL = LookingGlassCoreSDK-$(HOLOPLAYCORE_VERSION).tar.gz
HOLOPLAYCORE_URL = https://github.com/Looking-Glass/LookingGlassCoreSDK/archive/refs/tags/v$(HOLOPLAYCORE_VERSION).tar.gz
HOLOPLAYCORE_DYLIB_DIR = LookingGlassCoreSDK-$(HOLOPLAYCORE_VERSION)/HoloPlayCore/dylib
CLEAN_ALWAYS = src/lib
DISTCLEAN = $(HOLOPLAYCORE_TARBALL)

wheel install app-install: src/lib/libHoloPlayCore.dylib

# Extract all three platform libraries into src/lib/ as flat filenames.  Keying
# on an extracted library (rather than the tarball) ensures "make clean" + build
# re-extracts without needing the tarball to be re-downloaded.
src/lib/libHoloPlayCore.dylib: $(HOLOPLAYCORE_TARBALL)
	mkdir -p src/lib
	tar xf $(HOLOPLAYCORE_TARBALL) -C src/lib --strip-components=4 \
		$(HOLOPLAYCORE_DYLIB_DIR)/Win64/HoloPlayCore.dll \
		$(HOLOPLAYCORE_DYLIB_DIR)/linux/libHoloPlayCore.so \
		$(HOLOPLAYCORE_DYLIB_DIR)/macos/libHoloPlayCore.dylib

$(HOLOPLAYCORE_TARBALL):
	curl --silent --show-error --fail --location -o $@ $(HOLOPLAYCORE_URL)
