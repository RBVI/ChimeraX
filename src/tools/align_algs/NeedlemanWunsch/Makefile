TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/seqalign/align_algs/NeedlemanWunsch
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
endif

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

clean:
	rm -rf __pycache__
