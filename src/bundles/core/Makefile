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

BUNDLE_VERSION = 1.5
BUNDLE_VERSION := $(shell $(APP_PYTHON_EXE) mkbuildinfo -v $(TOP) $(BUNDLE_VERSION) $(BUILD_TYPE))

PYSRCS = $(filter-out setup.py,$(wildcard *.py)) src/buildinfo.py
DATA = src/colorbrewer.json

.PHONY: src/commands/_atomspec.py wheel src/buildinfo.py

build install reinstall:
	$(MAKE) app-install
	$(RSYNC) src/lib/* $(app_libdir)
	$(RSYNC) src/lib/* $(libdir)
	$(RSYNC) src/include/* $(app_includedir)
	$(RSYNC) src/include/* $(includedir)

app-install: clean src/buildinfo.py src/commands/_atomspec.py
	# newer versions of pip won't install over the same version number (which depends on the last commit
	# number), so to install something that hasn't been committed, uninstall first.
	$(APP_BOOTSTRAP_PIP) uninstall --yes ChimeraX-Core
	$(APP_PYTHON_EXE) -m pip install --no-build-isolation .

install-editable: clean src/buildinfo.py src/commands/_atomspec.py
	$(APP_BOOTSTRAP_PIP) uninstall --yes ChimeraX-Core
	$(APP_PYTHON_EXE) -m pip install --no-build-isolation --editable .

wheel: clean src/buildinfo.py
	$(APP_PYTHON_EXE) -m build --no-isolation --wheel

sync:
	rsync -av dist/*.whl $(build_prefix)/sync/

src/buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f src/buildinfo.py
	$(PYTHON_EXE) mkbuildinfo -b $(TOP) $(BUNDLE_VERSION) || (rm src/buildinfo.py ; exit 1)

src/commands/_atomspec.py: src/commands/_atomspec.peg
	$(APP_PYTHON_EXE) -m grako --name _atomspec -o $@ $<

clean:
	rm -rf __pycache__ build dist *.egg-info src/_serialize.cpp src/buildinfo.py \
		src/lib/* core_cpp/*/*.a core_cpp/*/*.o src/include src/*.so
