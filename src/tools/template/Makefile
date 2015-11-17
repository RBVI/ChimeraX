TOP = ../../..
include $(TOP)/mk/config.make

PKG_NAME = Replace with fully qualified Python package name
PKG_VERSION = Replace with package version number (PEP 440 style)
PKG_DESCRIPTION = Replace with short description of tool
# Example:
# PKG_NAME = chimera.cmd_line
# PKG_VERSION = 1.0
# PKG_DESCRIPTION = ChimeraX command line interface

# For many cases, only a single tool and a single command
# is provided by a package.  Variables below are designed
# for this case.  If you have multiple tools or commands,
# you need manually edit all the files.
TOOL_NAME = Replace with internal name
TOOL_PKG_NAME = Replace with tool Python package name
TOOL_DISPLAY_NAME = Replace with tool name to display to user
TOOL_CATEGORY = Replace with tool menu category name
TOOL_FILE_TYPES = Replace with file types
TOOL_SESSION_VERSIONS = Replace with minimum and maximum session versions
TOOL_CUSTOM_INIT = Replace with 'true' if custom initialization code is needed
TOOL_SYNOPSIS = Replace with synopsis for tool
COMMAND_NAME = Replace with one-word command name
# Example:
# TOOL_NAME = cmd_line
# TOOL_PKG_NAME = chimera.cmd_line
# TOOL_DISPLAY_NAME = Command Line Interface
# TOOL_CATEGORY = General
# TOOL_FILE_TYPES = 
# TOOL_SESSION_VERSIONS = 1,1
# TOOL_CUSTOM_INIT = false
# COMMAND_NAME = command_line

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,')
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
APP_PKG_DIR = $(APP_PYSITEDIR)/$(PKG_PATH)

PYSRCS = $(wildcard *.py)
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PYTHON_EXE) -m pip install -U --no-deps $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py timestamp
	$(PYTHON_EXE) setup.py bdist_wheel
	rm -rf $(PKG_NAME).egg-info

setup.py: setup.py.in Makefile
	# "TOOL_PKG_NAME" includes "PKG_NAME" so must be replaced first
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,#$(OS)#,,' \
		-e 's,#$(OSARCH)#,,' \
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
