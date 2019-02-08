include ../Makefile.bundle

DEVEL_ARGS = test false

wheel install app-install:	lib
clean: lib-clean

lib:
	$(MAKE) -C mmcif_cpp source

lib-clean:
	$(MAKE) -C mmcif_cpp clean
