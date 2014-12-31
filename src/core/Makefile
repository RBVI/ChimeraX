TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomstruct_cpp \
	   basegeom_cpp blob_cpp \
	   geometry graphics \
	   _mmcif \
	   ioutil_cpp logger_cpp \
	   pdb_cpp pdbio pseudobond_cpp \
	   readcif_cpp \
	   structaccess ui util_cpp

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	$(PYLINT) $(PYSRCS)

test:
	$(bindir)/python3
clean:
	rm -rf __pycache__

atomstruct.install: basegeom_cpp.install logger_cpp.install util_cpp.install \
	ioutil_cpp.install
blob_cpp.install: atomstruct.install
geometry.all: arrays_cpp.all
graphics.all: arrays_cpp.all
pdbio.install: atomstruct.install logger_cpp.install pdb_cpp.install \
	blob_cpp.install basegeom_cpp.install
pseudobond.install: basegeom_cpp.install
structaccess.install: atomstruct.install basegeom_cpp.install blob_cpp.install
