# The "make" targets are:
# 	wheel: build a Python wheel in "dist" directory.
# 	app-install: build wheel (if needed) and install in ChimeraX.
# 	test: run ChimeraX
# 	debug: run ChimeraX with debugging flag set
# 	clean: remove files used in building wheel
# 	distclean: remove files used in building wheel and license file

# These parameters may be changed as needed.

# ChimeraX bundle names must start with "ChimeraX_"
# to avoid clashes with package names in pypi.python.org.
# When uploaded to the ChimeraX toolshed, the bundle
# will be displayed without the ChimeraX- prefix.
BUNDLE_NAME = ChimeraX-Sample
BUNDLE_VERSION = 0.1
# ChimeraX bundles should only include packages
# that install as chimerax.package_name.
# General Python packages should be uploaded to
# pypi.python.org rather than the ChimeraX toolshed.
PKG_NAME = chimerax.sample
# Set PURE_PYTHON to 0 if bundle is platform-specific, 1 if pure Python
PURE_PYTHON = 0

# Define where ChimeraX is installed.
OS = $(patsubst CYGWIN_NT%,CYGWIN_NT,$(shell uname -s))
# CHIMERAX_APP is the ChimeraX install folder
ifeq ($(OS),CYGWIN_NT)
# Windows
CHIMERAX_APP = "/c/Program Files/ChimeraX.app"
CHIMERAX_APP = "/e/chimerax/ChimeraX.app"
endif
ifeq ($(OS),Darwin)
# Mac
CHIMERAX_APP = /Applications/ChimeraX.app
endif
ifeq ($(OS),Linux)
CHIMERAX_APP = /usr/local/bin/ChimeraX.app
endif

# ==================================================================
# Theoretically, no changes are needed below this line

# Platform-dependent settings.  Should not need fixing.
# For Windows, we assume Cygwin is being used.
ifeq ($(OS),CYGWIN_NT)
PYTHON_EXE = $(CHIMERAX_APP)/bin/python.exe
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
PYTHON_EXE = $(CHIMERAX_APP)/Contents/bin/python3.6
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
PYTHON_EXE = $(CHIMERAX_APP)/bin/python3.6
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif

BUNDLE_BASE_NAME = $(subst ChimeraX-,,$(BUNDLE_NAME))
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))
SOURCE = src
SRCS = $(SOURCE)/*.py $(SOURCE)/*.cpp

#
# Actual make dependencies!
#

wheel $(WHEEL): license.txt bundle_info.xml $(SRCS)
	$(CHIMERAX_EXE) --nogui --cmd "devel build . ; exit"

install app-install:	$(WHEEL)
	$(CHIMERAX_EXE) --nogui --cmd "devel install . ; exit"

uninstall app-uninstall:	$(WHEEL)
	$(CHIMERAX_EXE) --nogui --cmd "toolshed uninstall $(BUNDLE_BASE_NAME) ; exit"

test:
	$(CHIMERAX_EXE)

debug:
	$(CHIMERAX_EXE) --debug

clean:
	rm -rf __pycache__ build dist $(WHL_BNDL_NAME).egg-info setup.py
	rm -rf $(SOURCE)/__pycache__

license.txt:
	@echo Please specify your project licensing terms in file \"license.txt\"
	@echo The BSD license is included as file \"license.txt.bsd\"
	@echo The MIT license is included as file \"license.txt.mit\"
	@exit 1

bundle_info.xml:
	@echo Please create file \"bundle_info.xml\"
	@exit 1
