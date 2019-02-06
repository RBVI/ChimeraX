include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

DEVEL_ARGS = test false
CSRCS = search_cpp/imex.h search_cpp/search.cpp search_cpp/search.h \
	search_cpp/ast.pyx search_cpp/ast.pxd
CLEAN += src/*.$(PYMOD_EXT)
CLEAN_ALWAYS =	src/lib src/include \
		search_cpp/search.o* search_cpp/imex.h search_cpp/ast.cpp

wheel install app-install:	$(CSRCS)

search_cpp/imex.h:
	sed -e 's/LIBNAME/ATOM_SEARCH/' < $(includedir)/imex.i > $@
