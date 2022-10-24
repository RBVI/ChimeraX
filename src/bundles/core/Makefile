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
PYTHON ?=PYTHONNOUSERSITE=1 $(APP_PYTHON_EXE)
PIP ?=PYTHONNOUSERSITE=1 $(APP_BOOTSTRAP_PIP)

BUNDLE_VERSION = 1.6
BUNDLE_VERSION := $(shell $(PYTHON) mkbuildinfo -v $(TOP) $(BUNDLE_VERSION) $(BUILD_TYPE))

.PHONY: src/commands/_atomspec.py wheel src/buildinfo.py

install reinstall app-install: wheel
	# newer versions of pip won't install over the same version number (which depends on the last commit
	# number), so to install something that hasn't been committed, uninstall first.
	$(PIP) uninstall --yes ChimeraX-Core
	$(PYTHON) -m pip install dist/*.whl

build: install
	$(MAKE) sync

install-editable: clean src/buildinfo.py src/commands/_atomspec.py
	$(PIP) uninstall --yes ChimeraX-Core
ifeq ($(EDITABLE_MODE),strict)
	$(PYTHON) -m pip install --no-build-isolation --editable . --config-settings editable_mode=strict
else
	$(PYTHON) -m pip install --no-build-isolation --editable .
endif

wheel: clean src/buildinfo.py src/commands/_atomspec.py
	$(PYTHON) -m build --no-isolation --wheel

sync:
	rsync -av dist/*.whl $(build_prefix)/sync/

src/buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f src/buildinfo.py
	$(PYTHON) mkbuildinfo -b $(TOP) $(BUNDLE_VERSION) || (rm src/buildinfo.py ; exit 1)

src/commands/_atomspec.py: src/commands/_atomspec.peg
	$(PYTHON) -m grako --name _atomspec -o $@ $<

clean:
	rm -rf __pycache__ src/__pycache__ build dist *.egg-info src/_serialize.cpp \
		src/buildinfo.py src/commands/_atomspec.py src/toolshed/__pycache__ \
		src/lib/* core_cpp/*/*.a core_cpp/*/*.o src/include src/*.so \
		chutil_cpp logger_cpp ioutil_cpp mac_util_cpp toolshed commands \
		_serialize.cpp buildinfo.py setup.py
