# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make
CLEAN_ALWAYS =	src/lib src/include src/data atomic_cpp/pyinstance_cpp/imex*.h \
	atomic_cpp/element_cpp/imex.h atomic_cpp/atomstruct_cpp/imex.h

install: atomic_cpp/pyinstance_cpp/imex.map.h atomic_cpp/element_cpp/imex.h atomic_cpp/atomstruct_cpp/imex.h

atomic_cpp/atomstruct_cpp/imex.h:
	sed -e 's/LIBNAME/ATOMSTRUCT/' < $(includedir)/imex.i > $@

atomic_cpp/element_cpp/imex.h:
	sed -e 's/LIBNAME/ELEMENT/' < $(includedir)/imex.i > $@

atomic_cpp/pyinstance_cpp/imex.h:
	sed -e 's/LIBNAME/PYINSTANCE/' < $(includedir)/imex.i > $@

atomic_cpp/pyinstance_cpp/imex.map.h: atomic_cpp/pyinstance_cpp/imex.h
	sed -e 's/PYINSTANCE/PYINSTANCE_MAP/' < $< > $@
