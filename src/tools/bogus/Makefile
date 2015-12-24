TOP = ../../..
include $(TOP)/mk/config.make

PKG_NAME = chimerax.bogus
PKG_VERSION = 0.1
PKG_DESCRIPTION = Bogus Toolshed Demo Example

TOOL_NAME = bogus
TOOL_PKG_NAME = $(PKG_NAME)
TOOL_DISPLAY_NAME = Bogus Toolshed-demo Tool
TOOL_CATEGORY = General
TOOL_SYNOPSIS = Bogus tool for demonstrating toolshed functionality
TOOL_FILE_TYPES =
TOOL_SESSION_VERSIONS = 1,1
TOOL_CUSTOM_INIT = false
COMMAND_NAME = bogus

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,')
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
APP_PKG_DIR = $(APP_PYSITEDIR)/$(PKG_PATH)

PYSRCS = $(wildcard *.py)
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
# Do not put in app so we can install using toolshed
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
