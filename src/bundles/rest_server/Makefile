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

BUNDLE_NAME = ChimeraX-RestServer
BUNDLE_VERSION = 0.1
PKG_NAME = chimerax.rest_server

BASE_BNDL_NAME = $(subst ChimeraX-,,$(BUNDLE_NAME))

PYSRCS = $(wildcard src/*.py)
DATAFILES = src/static src/server.pem
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-py3-none-any.whl

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

wheel $(WHEEL): $(PYSRCS) $(DATAFILES)
	$(CHIMERAX_EXE) --nogui --cmd "devel build . ; exit"

install app-install: $(WHEEL)
	$(CHIMERAX_EXE) --nogui --cmd "devel install . user false; exit"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(CHIMERAX_EXE) --nogui --cmd "devel clean . ; exit"

lint:
	$(PYLINT) $(PYSRCS)

src/server.pem:	ssl.conf
	openssl req -new -x509 -keyout server.pem -out src/server.pem \
		-days 365 -nodes -config ssl.conf
