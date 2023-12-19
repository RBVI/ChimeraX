# The "make" targets are:
#    wheel: build a Python wheel in "dist" directory
#    install: build wheel and install
#    test: run ChimeraX
#    debug: run ChimeraX with debugging flag set
#    clean: remove files used in building wheel

# Theoretically, you should only need to change
# 1. the CHIMERAX_APP definition (or define in environment),
# 2. the "devel build" and "devel install" commands in the
#    "install" and "wheel" targets, (e.g., if you want to
#    install into ChimeraX.app instead of for the current
#    user, add "user false"), and
# 3. the "generated_files" and "clean_generated_files" targets
#    (e.g., if you need to generate files such as self-signed
#    certificates or data files from template expansion)

# Define where ChimeraX is installed.
OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
# CHIMERAX_APP is the ChimeraX install folder
# We use ?= to let CHIMERAX_APP environment variable override
ifeq ($(OS),CYGWIN_NT)
# Windows
CHIMERAX_APP ?= "c:/Program Files/ChimeraX.app"
endif
ifeq ($(OS),Darwin)
# Mac
CHIMERAX_APP ?= /Applications/ChimeraX.app
endif

# Platform-dependent settings.  Should not need fixing.
# For Windows, we assume Cygwin is being used.
ifeq ($(OS),CYGWIN_NT)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
CHIMERAX_EXE = $(shell which chimerax)
#CHIMERAX_EXE = $(shell which chimerax-daily)
endif
RUN = $(CHIMERAX_EXE) --nogui --exit --cmd

PYSRCS = $(wildcard src/*.py)
CSRCS = $(wildcard src/*.cpp)
SRCS = $(PYSRCS) $(CSRCS)

# If you want to install into ChimeraX.app, add "user false"
# to the "devel build" and "devel install" commands.
# By default, we install for just the current user.

install:	bundle_info.xml $(SRCS) generated_files
	$(RUN) "devel install . exit true"

wheel:	bundle_info.xml $(SRCS) generated_files
	$(RUN) "devel build . exit true"

test:
	for t in $(wildcard test*.cxc) $(wildcard test*.py);\
		do $(CHIMERAX_EXE) --exit --nogui $$t;\
	done

debug:
	$(CHIMERAX_EXE) --debug

clean:	clean_generated_files
	if [ -x $(CHIMERAX_EXE) ]; then \
		$(RUN) "devel clean . exit true" ; \
	else \
		rm -rf build dist *.egg-info src/__pycache__ ; \
	fi

pylint:
	$(CHIMERAX_EXE) -m flake8 $(filter %.py, $(SRCS))

# Modify targets below if you have files that need
# to be generated before building the bundle and/or
# removed when cleaning the bundle

generated_files:
	# Generate files

clean_generated_files:
	# Remove generated files
