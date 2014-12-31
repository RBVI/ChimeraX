TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomstruct_cpp \
	   basegeom_cpp blob_cpp \
	   geometry graphics \
	   _mmcif \
	   ioutil_cpp logger_cpp \
	   pdb_cpp pdbio pseudobond_cpp \
	   readcif_cpp \
	   structaccess toolshed ui util_cpp

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

PYSRCS = $(wildcard *.py)
CORE_VERSION = 2.0
WHEEL = dist/chimera.core-$(CORE_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	$(MAKE) app_install

app_install: wheel
	$(APP_PYTHON_EXE) -m pip install wheel distlib
	$(APP_PYTHON_EXE) -m pip install --ignore-installed $(WHEEL)

wheel:
	$(PYTHON_EXE) -m pip install wheel
	$(PYTHON_EXE) setup.py bdist_wheel

setup.py: setup.py.in
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		< setup.py.in > setup.py

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	$(PYLINT) $(PYSRCS)

test:
	$(bindir)/python3
clean:
	rm -rf __pycache__ build dist chimera.core.egg-info setup.py

atomstruct.install: basegeom_cpp.install logger_cpp.install util_cpp.install \
	ioutil_cpp.install
blob_cpp.install: atomstruct.install
geometry.all: arrays_cpp.all
graphics.all: arrays_cpp.all
pdbio.install: atomstruct.install logger_cpp.install pdb_cpp.install \
	blob_cpp.install basegeom_cpp.install
pseudobond.install: basegeom_cpp.install
structaccess.install: atomstruct.install basegeom_cpp.install blob_cpp.install
