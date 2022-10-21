# need to install C++ headers for subsequent builds
TOP = ../../..
include $(TOP)/mk/config.make
BUILD_AND_INSTALL = 1
include ../Makefile.bundle

CSRCS = connect_cpp/imex.h connect_cpp/connect.cpp connect_cpp/connect.h
CLEAN_ALWAYS =	src/lib src/include src/*.$(PYMOD_EXT) \
		connect_cpp/connect.o* \

wheel install app-install:	$(CSRCS)

connect_cpp/imex.h:
	sed -e 's/LIBNAME/PDB_CONNECT/' < $(TOP)/prereqs/imex/imex.i > $@

