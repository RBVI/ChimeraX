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

NO_SUBDIR_TEST=1
SUBDIRS	+= src/commands src/toolshed

include $(TOP)/mk/subdir.make

BUNDLE_NAME = ChimeraX-Core
PKG_NAME = chimerax.core

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,'g)
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifndef WIN32
BUILD_TOP = $(TOP)
else
BUILD_TOP := $(shell cygpath -m '$(TOP)' | sed 's,/,\\\\\\\\,g')
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
datadir := $(shell cygpath -m '$(datadir)' | sed 's,/,\\\\\\\\,g')
endif
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))

BUNDLE_VERSION = 1.5
BUNDLE_VERSION := $(shell $(PYTHON_EXE) src/mkbuildinfo -v $(BUILD_TOP) $(BUNDLE_VERSION) $(BUILD_TYPE))

GENSRCS = src/buildinfo.py
PYSRCS = $(filter-out src/setup.py,$(wildcard src/*.py)) $(GENSRCS)
DATA = src/colorbrewer.json
WHEEL = $(wildcard $(realpath dist)/*.whl)

all: $(PYOBJS)

install reinstall: dir all
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	# newer versions of pip won't install over the same version number (which depends on the last commit
	# number), so to install something that hasn't been committed, uninstall first.
	$(APP_BOOTSTRAP_PIP) uninstall --yes $(WHL_BNDL_NAME)
	$(APP_BOOTSTRAP_PIP) install dist/*.whl

.PHONY: wheel
wheel: pyproject.toml setup.py src/buildinfo.py
	rm -f $(WHEEL)
	$(PYTHON_EXE) -m build --wheel 
	rm -rf $(WHL_BNDL_NAME).egg-info

sync:
ifneq (,$(wildcard dist/*-py3-none-any.whl))
	rsync -av dist/*.whl $(build_prefix)/sync/python-only
else
	rsync -av dist/*.whl $(build_prefix)/sync/binary
endif

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
SETUP_SED_ARGS += -e 's,\# WINDEBUG,,'
endif
endif

.PHONY: pyproject.toml
pyproject.toml: pyproject.toml.in Makefile
	sed $(SETUP_SED_ARGS) < pyproject.toml.in > pyproject.toml

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

clean:
	rm -rf __pycache__ build dist *.egg-info pyproject.toml $(GENSRCS) src/_serialize.cpp

pylint:
	$(PYLINT) $(PYSRCS)

.PHONY: src/buildinfo.py
src/buildinfo.py: src/buildinfo.py.in src/mkbuildinfo
	rm -f src/buildinfo.py
	cd src; $(PYTHON_EXE) mkbuildinfo -b $(BUILD_TOP) $(BUNDLE_VERSION) || (rm src/buildinfo.py ; exit 1)
