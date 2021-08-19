# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

CSRCS = connect_cpp/imex.h connect_cpp/connect.cpp connect_cpp/connect.h
CLEAN_ALWAYS =	src/lib src/include src/*.$(PYMOD_EXT) \
		connect_cpp/connect.o* connect_cpp/imex.h \

wheel install app-install:	$(CSRCS)

connect_cpp/imex.h:
	sed -e 's/LIBNAME/PDB_CONNECT/' < $(includedir)/imex.i > $@

