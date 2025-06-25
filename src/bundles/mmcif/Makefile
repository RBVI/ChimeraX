include ../Makefile.bundle

build wheel install install-editable app-install uv-wheel uv-install:	lib
clean: lib-clean

lib:
	$(MAKE) -C mmcif_cpp source

lib-clean:
	$(MAKE) -C mmcif_cpp clean
