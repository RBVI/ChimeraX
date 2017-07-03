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

CHIMERAX_APP = ../../../ChimeraX.app

OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
ifeq ($(OS),CYGWIN_NT)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif

BUNDLE_NAME = ChimeraX-Alignments
BUNDLE_VERSION = 1.0
PKG_NAME = chimerax.seqalign

WHEEL = dist/$(subst -,_,$(BUNDLE_NAME))-$(BUNDLE_VERSION)-py3-none-any.whl
PYSRCS = $(wildcard src/*.py)

wheel $(WHEEL): $(PYSRCS) bundle_info.xml
	$(CHIMERAX_EXE) --nogui --cmd "dev build . ; exit"

install app-install: $(WHEEL)
	$(CHIMERAX_EXE) --nogui --cmd "toolshed install $(WHEEL) reinstall true user false; exit"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(CHIMERAX_EXE) --nogui --cmd "dev clean . ; exit"

lint:
	$(PYLINT) $(PYSRCS)
