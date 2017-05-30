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
# will be displayed without the ChimeraX_ prefix.
BUNDLE_NAME = ChimeraX_Sample
BUNDLE_BASE_NAME = $(BUNDLE_NAME:s/ChimeraX//)
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
CHIMERAX_APP = /d/chimerax/ChimeraX.app
endif
ifeq ($(OS),Darwin)
# Mac
CHIMERAX_APP = /usr/local/src/chimerax/ChimeraX.app
endif
ifeq ($(OS),Linux)
CHIMERAX_APP = /usr/local/src/chimerax/ChimeraX.app
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

# Paths used for constructing the installation "wheel"
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))
ifeq ($(PURE_PYTHON),1)
TAG = $(shell $(PYTHON_EXE) wheel_tag.py -p)
else
TAG = $(shell $(PYTHON_EXE) wheel_tag.py)
endif
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-$(TAG).whl
SOURCE = src
SRCS = $(SOURCE)/*.py $(SOURCE)/*.cpp

#
# Actual make dependencies!
#

wheel $(WHEEL): license.txt setup.py $(SRCS)
	$(PYTHON_EXE) setup.py --no-user-cfg build
	$(PYTHON_EXE) setup.py --no-user-cfg test
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel
	rm -rf $(WHL_BNDL_NAME).egg-info
	echo Distribution is in $(WHEEL)

app-install:	$(WHEEL)
	$(CHIMERAX_EXE) --nogui --cmd "toolshed install $(WHEEL) reinstall true ; exit"

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

setup.py: setup.py.in Makefile
	sed -e 's,BUNDLE_NAME,$(BUNDLE_NAME),' \
		-e 's,BUNDLE_VERSION,$(BUNDLE_VERSION),' \
		-e 's,PKG_NAME,$(PKG_NAME),' \
		< setup.py.in > setup.py

distclean:	clean
	rm -f license.txt
