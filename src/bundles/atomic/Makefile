include ../Makefile.bundle

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
