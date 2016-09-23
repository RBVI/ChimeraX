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

PKG_NAME = chimerax.volume_viewer
PKG_VERSION = 1.0
PKG_DESCRIPTION = User interface for density maps

# For many cases, only a single tool and a single command
# is provided by a package.  Variables below are designed
# for this case.  If you have multiple tools or commands,
# you need manually edit all the files.
TOOL_NAME = volume_viewer
TOOL_PKG_NAME = chimerax.volume_viewer
TOOL_DISPLAY_NAME = Map Viewer
TOOL_CATEGORY = Hidden
TOOL_FILE_TYPES =
TOOL_SESSION_VERSIONS = 1,1
TOOL_CUSTOM_INIT = false
TOOL_SYNOPSIS = Display density map settings
COMMAND_NAME = 

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,')
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
endif

PYSRCS = $(wildcard *.py)
DATAFILES = close.png hidden.png shown.png
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(DATAFILES) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PYTHON_EXE) -m pip install -U --no-deps $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py timestamp
	$(PYTHON_EXE) setup.py bdist_wheel
	rm -rf $(TOOL_NAME).egg-info

setup.py: setup.py.in Makefile
	# "TOOL_PKG_NAME" includes "PKG_NAME" so must be replaced first
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,SHLIB_DIR,$(shlibdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,TOOL_NAME,$(TOOL_NAME),' \
		-e 's,TOOL_PKG_NAME,$(TOOL_PKG_NAME),' \
		-e 's,TOOL_DISPLAY_NAME,$(TOOL_DISPLAY_NAME),' \
		-e 's/TOOL_CATEGORY/$(TOOL_CATEGORY)/' \
		-e 's/TOOL_FILE_TYPES/$(TOOL_FILE_TYPES)/' \
		-e 's/TOOL_SESSION_VERSIONS/$(TOOL_SESSION_VERSIONS)/' \
		-e 's/TOOL_CUSTOM_INIT/$(TOOL_CUSTOM_INIT)/' \
		-e 's/TOOL_SYNOPSIS/$(TOOL_SYNOPSIS)/' \
		-e 's/COMMAND_NAME/$(COMMAND_NAME)/' \
		-e 's,PKG_NAME,$(PKG_NAME),' \
		-e 's,PKG_VERSION,$(PKG_VERSION),' \
		-e 's,PKG_DESCRIPTION,$(PKG_DESCRIPTION),' \
		< setup.py.in > setup.py

timestamp: always
	echo "$(TOOL_NAME) $(PKG_VERSION)" >  timestamp

always:

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ build dist $(PKG_NAME).egg-info setup.py timestamp
