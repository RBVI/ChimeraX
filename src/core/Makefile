TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= ioutil pdb basegeom util pseudobond atomstruct blob pdbio \
	structaccess

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__

structaccess.install: atomstruct.install blob.install
pdbio.install: atomstruct.install pdb.install blob.install
