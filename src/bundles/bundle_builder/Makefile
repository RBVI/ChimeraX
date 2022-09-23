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

OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
ifeq ($(OS),CYGWIN_NT)
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

PYSRCS = $(wildcard src/*.py)

wheel: $(PYSRCS)
	$(PYTHON_EXE) -m build --no-isolation --wheel

sdist: $(PYSRCS)
	$(PYTHON_EXE) -m build --no-isolation --sdist

install app-install: wheel
	$(PYTHON_EXE) -m pip uninstall --yes ChimeraX-BundleBuilder
	$(PYTHON_EXE) -m pip install --no-build-isolation .
	$(MAKE) sync

install-editable:
	$(PYTHON_EXE) -m pip install --no-build-isolation --editable .

sync:
	rsync -av dist/*.whl $(TOP)/build/sync/

test:
	for t in $(wildcard test*.cxc) $(wildcard test*.py);\
		do $(CHIMERAX_EXE) --exit --nogui $$t;\
	done

debug:
	$(CHIMERAX_EXE) --debug

clean:
	rm -rf src/__pycache__ build dist *.egg-info

pylint:
	$(PYLINT) $(PYSRCS)
