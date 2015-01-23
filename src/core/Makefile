TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomstruct_cpp \
	   basegeom_cpp blob_cpp \
	   geometry graphics \
	   _mmcif \
	   ioutil_cpp logger_cpp \
	   pdb_cpp pdbio pseudobond_cpp \
	   readcif_cpp \
	   structaccess toolshed ui chutil_cpp

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core

PYSRCS = $(wildcard *.py)
CORE_VERSION = 2.0
WHEEL = dist/chimera.core-$(CORE_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PYTHON_EXE) -m pip install --no-deps $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse)

wheel: setup.py timestamp always
	$(PYTHON_EXE) setup.py bdist_wheel

setup.py: setup.py.in
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,CORE_VERSION,$(CORE_VERSION),' \
		< setup.py.in > setup.py

timestamp:
	echo "chimera.core" > timestamp

always:

lint:
	$(PYLINT) $(PYSRCS)

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

test:
	$(APP_PYTHON_EXE)

clean:
	rm -rf __pycache__ build dist chimera.core.egg-info setup.py

_appdirs.install: appdirs_cpp.install
atomstruct_cpp.install: basegeom_cpp.install logger_cpp.install \
	chutil_cpp.install ioutil_cpp.install pseudobond_cpp.install
blob_cpp.install: atomstruct_cpp.install
geometry.install: arrays_cpp.install
graphics.install: arrays_cpp.install
_mmcif.install: readcif_cpp.install atomstruct_cpp.install blob_cpp.install
pdbio.install: atomstruct_cpp.install logger_cpp.install pdb_cpp.install \
	blob_cpp.install basegeom_cpp.install
pseudobond_cpp.install: basegeom_cpp.install
structaccess.install: atomstruct_cpp.install basegeom_cpp.install blob_cpp.install
