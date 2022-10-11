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

CHIMERAX_APP = $(wildcard $(TOP)/ChimeraX*.app)

ifeq ($(OS),Windows)
PYTHON_BIN = $(CHIMERAX_APP)/bin/python.exe
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
PYTHON_BIN = $(CHIMERAX_APP)/Contents/bin/python$(PYTHON_VERSION)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
PYTHON_BIN = $(CHIMERAX_APP)/bin/python$(PYTHON_VERSION)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif
PYTHON_EXE = $(PYTHON_BIN) -I
PYLINT = $(PYTHON_EXE) -m flake8

BUNDLE_NAME = ChimeraX-ToolshedUtils
BUNDLE_VERSION = 1.2.1
PKG_NAME = chimerax.toolshed_utils

BASE_BNDL_NAME = $(subst ChimeraX-,,$(BUNDLE_NAME))
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))

PYSRCS = $(wildcard src/*.py)
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-py$(PYVER_NODOT)-none-any.whl

wheel $(WHEEL): setup.py $(PYSRCS)
	$(PYTHON_EXE) setup.py --no-user-cfg build
	# $(PYTHON_EXE) setup.py --no-user-cfg test
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel --python-tag py$(PYVER_NODOT)
	rm -rf $(WHL_BNDL_NAME).egg-info
	$(MAKE) sync

install app-install: $(WHEEL)
	$(PYTHON_EXE) -m pip uninstall --yes $(BUNDLE_NAME)
	$(PYTHON_EXE) -m pip install $(WHEEL)

sync:
	rsync -av dist/*.whl $(TOP)/build/sync/

build: install
	# unlike normal bundles, install as well
	$(MAKE) sync

test:
	for t in $(wildcard test*.cxc) $(wildcard test*.py);\
		do $(CHIMERAX_EXE) --exit --nogui $$t;\
	done

debug:
	$(CHIMERAX_EXE) --debug

clean:
	rm -rf src/__pycache__ build dist $(WHL_BNDL_NAME).egg-info setup.py

pylint:
	$(PYLINT) $(PYSRCS)

setup.py: setup.py.in Makefile
	sed \
		-e 's,BUNDLE_NAME,$(BUNDLE_NAME),' \
		-e 's,BUNDLE_VERSION,$(BUNDLE_VERSION),' \
		-e 's,PKG_NAME,$(PKG_NAME),' \
		< setup.py.in > setup.py
