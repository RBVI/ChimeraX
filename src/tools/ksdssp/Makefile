TOP = ../../..
include $(TOP)/mk/config.make

PKG_NAME = chimerax.ksdssp
PKG_VERSION = 1.0
PKG_DESCRIPTION = Compute\/assign Kabsch & Sander secondary structure

TOOL_NAME = ksdssp
TOOL_PKG_NAME = $(PKG_NAME)
TOOL_DISPLAY_NAME = Ksdssp
TOOL_CATEGORY = Hidden
TOOL_FILE_TYPES =
TOOL_SESSION_VERSIONS = 1,1
TOOL_CUSTOM_INIT = false
TOOL_SYNOPSIS = $(PKG_DESCRIPTION)
COMMAND_NAME = ss_assign

PKG_DIR		= $(PYSITEDIR)/chimerax/ksdssp
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
endif

PYMOD_NAME	= _ksdssp
SRCS		= ksdssp.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS)
ifdef WIN32
LIBS		+= $(PYTHON_LIB) libatomstruct.lib liblogger.lib
else
LIBS		+= $(PYTHON_LIB) -latomstruct -llogger
endif

PYSRCS = __init__.py ksdssp.py
WHEEL = dist/$(PKG_NAME)-$(PKG_VERSION)-py3-none-any.whl

all: $(PYMOD) $(PYOBJS)

install: all dir
	$(RSYNC) $(PYMOD) $(PYSRCS) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PYSRCS) $(PKG_DIR)
endif
endif
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
	rm -rf __pycache__ build dist $(PKG_NAME).egg-info setup.py timestamp $(PYMOD) $(PYOBJS)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

