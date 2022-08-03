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

TOP = ../../..
include $(TOP)/mk/config.make

NO_SUBDIR_TEST=1
SUBDIR_BUILD=1
SUBDIRS	+= chutil_cpp commands \
	   ioutil_cpp logger_cpp \
	   pysupport_cpp toolshed

ifeq ($(OS),Darwin)
SUBDIRS += mac_util_cpp
endif

include $(TOP)/mk/subdir.make

BUNDLE_NAME = ChimeraX-Core
PKG_NAME = chimerax.core

# PKG_PATH = chimerax/core
PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,'g)
# PKG_DIR = $(TOP)/build/lib/python3.9/site-packages/chimerax/core
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
BUNDLE_VERSION := $(shell $(APP_PYTHON_EXE) mkbuildinfo -v $(BUILD_TOP) $(BUNDLE_VERSION) $(BUILD_TYPE))

GENSRCS = buildinfo.py
PYSRCS = $(filter-out setup.py,$(wildcard *.py)) $(GENSRCS)
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-cp$(PYVER_NODOT)-cp$(PYVER_NODOT)$(PYTHON_ABI)-$(BIN_PLATFORM).whl
DATA = colorbrewer.json

# So instead let's build everything from scratch with the PEP 517 bundle builder. 
# First, we'll copy all the header files over into the directories they're already
# going into, so that they can be used by downstream packages that haven't been
# converted yet. 
#
# Then, we'll build ChimeraX-Core.
#
# Lastly, we'll copy the built objects over, again so that downstream packages that
# haven't yet been converted can still build. 

all: $(PYOBJS)

build install reinstall: dir all
	# Install the sources and data to pysitedir;
	# we install by hand so that we don't have to install
	# with the python in the build directory?
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	# then if the app prefix is made, we install there too
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	# newer versions of pip won't install over the same version number (which depends on the last commit
	# number), so to install something that hasn't been committed, uninstall first.
	$(APP_BOOTSTRAP_PIP) uninstall --yes $(WHL_BNDL_NAME)
	$(APP_BOOTSTRAP_PIP) install $(WHEEL)
	# then we send it to /build/sync
	$(MAKE) sync

.PHONY: wheel
wheel: setup.py
	rm -f $(WHEEL)
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel --plat-name $(BIN_PLATFORM)
	rm -rf $(WHL_BNDL_NAME).egg-info

sync:
	rsync -av dist/*.whl $(build_prefix)/sync/

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

.PHONY: setup.py
setup.py: setup.py.in Makefile
	sed $(SETUP_SED_ARGS) < setup.py.in > setup.py

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

clean:
	rm -rf __pycache__ build dist $(WHL_BNDL_NAME).egg-info setup.py $(GENSRCS) _serialize.cpp

pylint:
	$(PYLINT) $(PYSRCS)

.PHONY: buildinfo.py
buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f buildinfo.py
	$(PYTHON_EXE) mkbuildinfo -b $(BUILD_TOP) $(BUNDLE_VERSION) || (rm buildinfo.py ; exit 1)
