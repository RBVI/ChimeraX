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

PKG_NAME = chimerax.shortcuts
PKG_VERSION = 1.0
PKG_DESCRIPTION = Keboard shortcuts

# For many cases, only a single tool and a single command
# is provided by a package.  Variables below are designed
# for this case.  If you have multiple tools or commands,
# you need manually edit all the files.
TOOL_NAME = shortcuts
TOOL_PKG_NAME = chimerax.shortcuts
TOOL_DISPLAY_NAME = Molecule Display
TOOL_CATEGORY = Toolbars
TOOL_FILE_TYPES =
TOOL_SESSION_VERSIONS = 1,1
TOOL_CUSTOM_INIT = false
TOOL_SYNOPSIS = Buttons and keyboard shortcuts for common commands
COMMAND_NAME = ks

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,')
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
endif

PYSRCS = __init__.py tool.py shortcuts.py
DATAFILES = icons
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(DATAFILES) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PIP) install -U $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel
	rm -rf $(TOOL_NAME).egg-info

setup.py: setup.py.in Makefile
	# "TOOL_PKG_NAME" includes "PKG_NAME" so must be replaced first
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,SHLIB_DIR,$(shlibdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		< setup.py.in > setup.py

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ build dist $(PKG_NAME).egg-info setup.py
