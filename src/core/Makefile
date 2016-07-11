TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomic \
	   chutil_cpp commands devices \
	   geometry graphics \
	   ioutil_cpp logger_cpp \
	   map pysupport_cpp surface toolshed ui \
	   webservices

ifeq ($(OS),Darwin)
SUBDIRS += mac_util_cpp
endif

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax/core
ifdef WIN32
TOP := $(shell cygpath -m '$(TOP)' | sed 's,/,\\\\\\\\,g')
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
datadir := $(shell cygpath -m '$(datadir)' | sed 's,/,\\\\\\\\,g')
endif

GENSRCS = buildinfo.py
PYSRCS = $(wildcard *.py) $(GENSRCS)
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

SETUP_SED_ARGS = -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,SHLIB_DIR,$(shlibdir),' \
		-e 's,DATA_DIR,$(datadir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,\#$(OS)\#,,' \
		-e 's,\#$(OSARCH)\#,,' \
		-e 's,CORE_VERSION,$(CORE_VERSION),'
ifdef WIN32
ifdef DEBUG
SETUP_SED_ARGS += -e 's,\#WINDEBUG,,'
endif
endif

setup.py: setup.py.in
	sed $(SETUP_SED_ARGS) < setup.py.in > setup.py

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

buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f buildinfo.py
	$(PYTHON_EXE) mkbuildinfo $(TOP) > buildinfo.py || rm -f buildinfo.py

_appdirs.install: appdirs_cpp.install
atomic.install: arrays_cpp.install chutil_cpp.install ioutil_cpp.install \
	logger_cpp.install pysupport_cpp.install
geometry.install: arrays_cpp.install
graphics.install: arrays_cpp.install
map.install: arrays_cpp.install
surface.install: arrays_cpp.install
