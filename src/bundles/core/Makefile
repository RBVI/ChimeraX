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

build install reinstall:
	#$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	$(MAKE) app-install
	#$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

app-install: clean src/buildinfo.py wheel
	# newer versions of pip won't install over the same version number (which depends on the last commit
	# number), so to install something that hasn't been committed, uninstall first.
	$(APP_BOOTSTRAP_PIP) uninstall --yes ChimeraX-Core
	$(APP_PYTHON_EXE) -m pip install dist/*.whl

.PHONY: wheel
wheel: clean src/buildinfo.py
	$(APP_PYTHON_EXE) -m build --no-isolation --wheel

sync:
	rsync -av dist/*.whl $(build_prefix)/sync/

clean:
	rm -rf __pycache__ build dist *.egg-info src/_serialize.cpp src/buildinfo.py \
		src/lib/* core_cpp/*/*.a core_cpp/*/*.o src/include

pylint:
	$(PYLINT) $(PYSRCS)

.PHONY: src/buildinfo.py
src/buildinfo.py: buildinfo.py.in mkbuildinfo
	rm -f src/buildinfo.py
	$(PYTHON_EXE) mkbuildinfo -b $(TOP) $(BUNDLE_VERSION) || (rm src/buildinfo.py ; exit 1)

#all: $(PYOBJS)
#
#build install: all dir
#	$(RSYNC) $(PYSRCS) $(PKG_DIR)
#
#dir:
#	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
#	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi
#
#.SUFFIXES: .peg
#.peg.py:
#	$(PYTHON_EXE) -m grako --name $* -o $@ $^
#
#test_peg: _atomspec.py
#	$(PYTHON_EXE) _atomspec.py --trace atomspec_test.in atom_specifier
