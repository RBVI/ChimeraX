TOP = ../../../..
include $(TOP)/mk/config.make

SUBDIRS = MDToolsMarch97

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax/md_crds/dcd
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
