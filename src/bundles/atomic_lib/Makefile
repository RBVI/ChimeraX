# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle

CLEAN += src/*.$(PYMOD_EXT)
CLEAN_ALWAYS += src/data src/lib src/include

wheel install app-install:	lib

lib:
	$(MAKE) -C atomic_cpp install

clean:	atomic_cpp.clean

atomic_cpp.clean:
	$(MAKE) -C atomic_cpp clean
