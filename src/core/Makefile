TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp arrays_cpp atomstruct basegeom blob \
	geometry graphics ioutil logger_cpp pdb pdbio  pseudobond structaccess ui util

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	$(PYTHON_EXE) -m flake8 $(PYSRCS)

clean:
	rm -rf __pycache__

atomstruct.install: basegeom.install logger_cpp.install util.install
blob.install: atomstruct.install
geometry.install: arrays_cpp.install
graphics.install: arrays_cpp.install
pdbio.install: atomstruct.install logger_cpp.install pdb.install blob.install \
	basegeom.install
pseudobond.install: basegeom.install
structaccess.install: atomstruct.install basegeom.install blob.install
