# need to install C++ headers for subsequent builds
TOP = ../../..
include $(TOP)/mk/config.make
BUILD_AND_INSTALL = 1
include ../Makefile.bundle

CSRCS = search_cpp/ast.pyx search_cpp/ast.pxd
CLEAN += src/*.$(PYMOD_EXT)
CLEAN_ALWAYS = search_cpp/ast.cpp

wheel install app-install:	$(CSRCS)
