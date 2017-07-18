# The "make" targets are:
# 	wheel: build a Python wheel in "dist" directory.
# 	install: build wheel (if needed) and install in ChimeraX.
# 	test: run ChimeraX
# 	debug: run ChimeraX with debugging flag set
# 	clean: remove files used in building wheel

CHIMERAX_APP = ../../../ChimeraX.app

# Platform-dependent settings.  Should not need fixing.
# For Windows, we assume Cygwin is being used.
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

SRCS = $(wildcard src/*.py)

wheel: bundle_info.xml $(SRCS)
	$(CHIMERAX_EXE) --nogui --cmd "devel build . ; exit"

install app-install:	 bundle_info.xml $(SRCS)
	$(CHIMERAX_EXE) --nogui --cmd "devel install . user false; exit"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(CHIMERAX_EXE) --nogui --cmd "devel clean . ; exit"
	rm -f bundle_info.xml

pylint:
	$(CHIMERAX_EXE) -m flake8 $(filter %.py, $(SRCS))

bundle_info.xml:	bundle_info.xml.in
	$(CHIMERAX_EXE) --nogui make_selectors.py
