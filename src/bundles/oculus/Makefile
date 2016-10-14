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

PKG_NAME = chimerax.oculus
PKG_VERSION = 1.0
PKG_DESCRIPTION = Oculus Rift support

TOOL_NAME = Oculus Rift
TOOL_PKG_NAME = chimerax.oculus
TOOL_DISPLAY_NAME = Oculus Rift
TOOL_CATEGORY = Hidden
TOOL_FILE_TYPES =
TOOL_SESSION_VERSIONS = 1,1
TOOL_CUSTOM_INIT = false
TOOL_SYNOPSIS = View models with Oculus Rift
COMMAND_NAME = oculus

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,'g)
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
endif

PYSRCS = __init__.py oculuscmd.py track.py window.py
DATAFILES =
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

PYMOD_NAME = _oculus
SRCS	= oculus.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
OCSDK = $(libdir)/LibOVR
OCINC = -I$(OCSDK)/Include
DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS) $(OCINC)
#LIBS	+= $(PYTHON_LIB) $(OCSDK)/Lib/Mac/Release/libovr.a
LIBS	+= $(PYTHON_LIB) $(OCSDK)/Lib/Mac/Debug/libovr.a
LIBS	+= -framework Cocoa -framework IOKit -framework OpenGL
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python

# libovr is compiled without rtti and without exceptions
OPT = -fno-rtti -fno-exceptions -g -Wall

ifeq ($(shell test -e $(OCSDK) && echo found),found)
	HAVE_OCULUS_SDK = 1
else
	PYMOD = 
	PYOBJS = 
endif

all: $(PYMOD) $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(PYMOD) $(DATAFILES) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PIP) install -U $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel
	rm -rf $(PKG_NAME).egg-info

ifdef HAVE_OCULUS_SDK

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix _oculus.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PYMOD)
endif

endif

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

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ build dist $(PKG_NAME).egg-info setup.py $(PYMOD) $(OBJS)
