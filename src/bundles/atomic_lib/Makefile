# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

CSRCS = atomic_cpp/load_libs.cpp
CLEAN_ALWAYS =	src/lib src/include src/data atomic_cpp/load_libs.o*

wheel install app-install:	lib

lib:
	$(MAKE) -C atomic_cpp install

clean:	atomic_cpp.clean

atomic_cpp.clean:
	$(MAKE) -C atomic_cpp clean
