TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= atomstruct basegeom blob cpp_appdirs cpp_logger \
	ioutil pdb pdbio  pseudobond structaccess ui util

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
	python3 -m flake8 $(PYSRCS)

clean:
	rm -rf __pycache__

atomstruct.install: cpp_logger.install
structaccess.install: atomstruct.install blob.install
pdbio.install: atomstruct.install pdb.install blob.install
