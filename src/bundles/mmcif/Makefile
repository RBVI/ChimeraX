# Cython modules that use only Python's C API
PREBUILD_EXTENSIONS = mmcif mmcif_write

include ../Makefile.bundle

build wheel install install-editable app-install uv-wheel uv-install:	lib
clean: lib-clean

prebuild: lib

lib:
	$(MAKE) -C mmcif_cpp source

lib-clean:
	$(MAKE) -C mmcif_cpp clean
