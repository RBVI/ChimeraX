include ../Makefile.bundle

CLEAN_ALWAYS += src/NeedlemanWunsch/__pycache__
ifeq ($(OS),CYGWIN_NT)
	CLEAN_ALWAYS += src/support.obj src/_[sn]w*.pyd src/libalign_algs.*
endif
ifeq ($(OS),Darwin)
	CLEAN_ALWAYS += src/support.o src/_[sn]w*.so src/libalign_algs.*
endif
ifeq ($(OS),Linux)
	CLEAN_ALWAYS += src/support.o src/_[sn]w*.so src/libalign_algs.*
endif
