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

TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/commands
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/commands

GENSRCS = _atomspec.py
PYSRCS = $(wildcard *.py) $(GENSRCS)

all: $(PYOBJS)

build install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

.SUFFIXES: .peg

.peg.py:
	$(PYTHON_EXE) -m grako --name $* -o $@ $^

test_peg:	_atomspec.py
	$(PYTHON_EXE) _atomspec.py --trace atomspec_test.in atom_specifier

clean:
	rm -rf __pycache__ $(GENSRCS)

pylint:
	$(PYLINT) $(PYSRCS)
