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
RUN = $(CHIMERAX_EXE) --nogui --exit --cmd

SRCS = $(wildcard src/*.py)

wheel:	bundle_info.xml $(SRCS)
	$(RUN) "devel build . exit true"

install app-install:	bundle_info.xml $(SRCS)
	$(RUN) "devel install . user false exit true"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	$(RUN) "devel clean . exit true"

pylint:
	$(CHIMERAX_EXE) -m flake8 $(filter %.py, $(SRCS))
