include ../Makefile.bundle

ifeq ($(OS),CYGWIN_NT)
	CLEAN += src/support.obj src/_[sn]w*.pyd src/libalign_algs.*
endif
ifeq ($(OS),Darwin)
	CLEAN += src/support.o src/_[sn]w*.so src/libalign_algs.*
endif
ifeq ($(OS),Linux)
	CLEAN += src/support.o src/_[sn]w*.so src/libalign_algs.*
endif
