# need to install C++ headers for subsequent builds
TOP = ../../..
include $(TOP)/mk/config.make
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
CLEAN_ALWAYS =	src/lib src/include src/data atomic_cpp/*/*.$(OBJ_EXT) \
		atomic_cpp/atomstruct_cpp/tmpl/*.$(OBJ_EXT)

install: atomic_cpp/pyinstance_cpp/imex.map.h atomic_cpp/element_cpp/imex.h atomic_cpp/atomstruct_cpp/imex.h

atomic_cpp/atomstruct_cpp/imex.h:
	sed -e 's/LIBNAME/ATOMSTRUCT/' < $(TOP)/prereqs/imex/imex.i > $@

atomic_cpp/element_cpp/imex.h:
	sed -e 's/LIBNAME/ELEMENT/' < $(TOP)/prereqs/imex/imex.i > $@

atomic_cpp/pyinstance_cpp/imex.h:
	sed -e 's/LIBNAME/PYINSTANCE/' < $(TOP)/prereqs/imex/imex.i > $@

atomic_cpp/pyinstance_cpp/imex.map.h: atomic_cpp/pyinstance_cpp/imex.h
	sed -e 's/PYINSTANCE/PYINSTANCE_MAP/' < $< > $@
