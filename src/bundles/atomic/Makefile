include ../Makefile.bundle

DEVEL_ARGS = test false
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
