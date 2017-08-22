# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	+= appdirs_cpp _appdirs arrays_cpp atomic \
	   chutil_cpp commands \
	   geometry graphics \
	   ioutil_cpp logger_cpp \
	   map pysupport_cpp surface toolshed ui \
	   webservices

ifeq ($(OS),Darwin)
SUBDIRS += mac_util_cpp
endif

include $(TOP)/mk/subdir.make

BUNDLE_NAME = ChimeraX-Core
BUNDLE_VERSION = 0.1
PKG_NAME = chimerax.core

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,'g)
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifdef WIN32
TOP := $(shell cygpath -m '$(TOP)' | sed 's,/,\\\\\\\\,g')
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
datadir := $(shell cygpath -m '$(datadir)' | sed 's,/,\\\\\\\\,g')
endif
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))

GENSRCS = buildinfo.py
PYSRCS = $(filter-out setup.py,$(wildcard *.py)) $(GENSRCS)
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-py3-none-$(BIN_PLATFORM).whl

.FORCE: wheel

all: $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_BOOTSTRAP_PIP) install -U $(WHEEL)

wheel: setup.py
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel --plat-name $(BIN_PLATFORM)
	rm -rf $(WHL_BNDL_NAME).egg-info

SETUP_SED_ARGS = -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,SHLIB_DIR,$(shlibdir),' \
		-e 's,DATA_DIR,$(datadir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,\#$(OS)\#,,' \
		-e 's,\#$(OSARCH)\#,,' \
		-e 's,BUNDLE_NAME,$(BUNDLE_NAME),' \
		-e 's,BUNDLE_VERSION,$(BUNDLE_VERSION),' \
		-e 's,PKG_NAME,$(PKG_NAME),'
ifdef WIN32
ifdef DEBUG
SETUP_SED_ARGS += -e 's,\#WINDEBUG,,'
endif
endif

setup.py: setup.py.in
	sed $(SETUP_SED_ARGS) < setup.py.in > setup.py

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

clean:
	rm -rf __pycache__ build dist $(WHL_BNDL_NAME).egg-info setup.py $(GENSRCS)

pylint:
	$(PYLINT) $(PYSRCS)

.PHONY: buildinfo.py
buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f buildinfo.py
	$(PYTHON_EXE) mkbuildinfo $(TOP) $(BUNDLE_VERSION) > buildinfo.py || rm -f buildinfo.py

_appdirs.install: appdirs_cpp.install
atomic.install: arrays_cpp.install chutil_cpp.install ioutil_cpp.install \
	logger_cpp.install pysupport_cpp.install
geometry.install: arrays_cpp.install
graphics.install: arrays_cpp.install
map.install: arrays_cpp.install
surface.install: arrays_cpp.install
