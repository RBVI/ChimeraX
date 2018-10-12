include ../Makefile.bundle

DEVEL_ARGS = test false

wheel install app-install:	lib
clean: lib-clean

lib:
	cd mmcif_cpp; $(MAKE) source

lib-clean:
	cd mmcif_cpp; $(MAKE) clean
