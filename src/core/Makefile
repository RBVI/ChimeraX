TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomic \
	   chutil_cpp commands devices \
	   geometry graphics \
	   ioutil_cpp logger_cpp \
	   map surface toolshed ui \
	   webservices

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core

PYSRCS = $(wildcard *.py)
CORE_VERSION = 0.1
WHEEL = dist/chimerax.core-$(CORE_VERSION)-py3-none-any.whl

.FORCE: wheel

all: $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PYTHON_EXE) -m pip install -U --no-deps $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse)

wheel: setup.py timestamp
	$(PYTHON_EXE) setup.py bdist_wheel

setup.py: setup.py.in
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		-e 's,DATA_DIR,$(datadir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,#$(OS)#,,' \
		-e 's,#$(OSARCH)#,,' \
		-e 's,CORE_VERSION,$(CORE_VERSION),' \
		< setup.py.in > setup.py

timestamp:
	echo "chimerax.core" > timestamp

lint:
	$(PYLINT) $(PYSRCS)

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

test:
	$(APP_PYTHON_EXE)

clean:
	rm -rf __pycache__ build dist chimerax.core.egg-info setup.py timestamp $(GENSRCS)

_appdirs.install: appdirs_cpp.install
atomic.install: arrays_cpp.install chutil_cpp.install ioutil_cpp.install logger_cpp.install
geometry.install: arrays_cpp.install
graphics.install: arrays_cpp.install
map.install: arrays_cpp.install
surface.install: arrays_cpp.install
