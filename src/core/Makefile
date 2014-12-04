TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp arrays_cpp atomstruct basegeom blob \
	geometry graphics ioutil logger_cpp pdb pdbio  pseudobond structaccess ui util

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

atomstruct.install: basegeom.install logger_cpp.install util.install \
	ioutil.install
blob.install: atomstruct.install
geometry.all: arrays_cpp.all
graphics.all: arrays_cpp.all
pdbio.install: atomstruct.install logger_cpp.install pdb.install blob.install \
	basegeom.install
pseudobond.install: basegeom.install
structaccess.install: atomstruct.install basegeom.install blob.install
