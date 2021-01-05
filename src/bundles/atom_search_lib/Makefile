# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

CSRCS = search_cpp/imex.h search_cpp/search.cpp search_cpp/search.h search_cpp/load_libs.cpp
CLEAN_ALWAYS =	src/lib src/include search_cpp/search.o* search_cpp/load_libs.o* search_cpp/imex.h

wheel install app-install:	$(CSRCS)

search_cpp/imex.h:
	sed -e 's/LIBNAME/ATOM_SEARCH/' < $(includedir)/imex.i > $@
