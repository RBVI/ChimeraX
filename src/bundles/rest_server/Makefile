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

PYSRCS = $(wildcard src/*.py)
DATAFILES = src/static src/server.pem
SRCS = $(PYSRCS) $(DATAFILES)

wheel:	bundle_info.xml $(SRCS)
	$(CHIMERAX_EXE) --nogui --cmd "devel build . ; exit"

install app-install:	bundle_info.xml $(SRCS)
	$(CHIMERAX_EXE) --nogui --cmd "devel install . user false; exit"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(CHIMERAX_EXE) --nogui --cmd "devel clean . ; exit"
