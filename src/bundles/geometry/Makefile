# needed to get import of chimerax.atomic to work
BUILD_AND_INSTALL = 1
include ../Makefile.bundle

CLEAN_ALWAYS +=	src/_geometry*.$(PYMOD_EXT)

ifeq ($(OS),Linux)
# hack around linker giving us a RUNPATH when we asked for RPATH

install: make-rpath

make-rpath: wheel
	$(TOP)/build/bin/chrpath -C $$(find . -name \*.so)
	rm dist/*.whl
	$(MAKE) wheel
endif
