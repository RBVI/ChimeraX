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

PYTHON ?= PYTHONNOUSERSITE=1 $(APP_PYTHON_EXE)
PYLINT = $(PYTHON) -m flake8

PYSRCS = $(wildcard src/*.py)

wheel: $(PYSRCS)
	$(PYTHON) -m build --no-isolation --wheel

sdist: $(PYSRCS)
	$(PYTHON) -m build --no-isolation --sdist

install app-install: wheel
	$(PYTHON) -m pip uninstall --yes ChimeraX-BundleBuilder
	$(PYTHON) -m pip install --no-build-isolation .

build: install
	$(MAKE) sync

install-editable:
	$(PYTHON) -m pip install --no-build-isolation --editable .

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
