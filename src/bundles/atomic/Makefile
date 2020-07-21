# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle

CLEAN_ALWAYS += src/data src/lib src/include
ifeq ($(OS),CYGWIN_NT)
CLEAN_ALWAYS += src/libmolc.dll src/cymol.pyd
endif
ifeq ($(OS),Darwin)
CLEAN_ALWAYS += src/libmolc.dylib src/cymol.so
endif
ifeq ($(OS),Linux)
CLEAN_ALWAYS += src/libmolc.so src/cymol.so
endif

wheel install app-install:	lib

lib:
	$(MAKE) -C atomic_cpp

clean:	atomic_cpp.clean

atomic_cpp.clean:
	$(MAKE) -C atomic_cpp clean
