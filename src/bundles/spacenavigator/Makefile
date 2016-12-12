# Fix all parts that say FIXME
#
TOP = ../../..
include $(TOP)/mk/config.make

BUNDLE_NAME = ChimeraX-SpaceNavigator
BUNDLE_VERSION = 1.0
PKG_NAME = chimerax.spacenavigator

PKG_PATH = $(shell echo $(PKG_NAME) | sed 's,\.,/,'g)
PKG_DIR = $(PYSITEDIR)/$(PKG_PATH)
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
shlibdir := $(shell cygpath -m '$(shlibdir)' | sed 's,/,\\\\\\\\,g')
datadir := $(shell cygpath -m '$(datadir)' | sed 's,/,\\\\\\\\,g')
endif
WHL_BNDL_NAME = $(subst -,_,$(BUNDLE_NAME))

PYSRCS = $(filter-out setup.py,$(wildcard *.py))
DATAFILES = 
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-py3-none-$(BIN_PLATFORM).whl

ifeq ($(OS),Darwin)
SNFW = $(wildcard /Library/Frameworks/3DconnexionClient.framework)

ifneq (,$(SNFW))
HAVE_PYMOD = 1
PYMOD_NAME = _spacenavigator
SRCS	= spacenav.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
SNINC	= -I$(SNFW)/Versions/Current/Headers
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS) $(SNINC)
LIBS	+= $(PYTHON_LIB) -F/Library/Frameworks -framework 3DconnexionClient
PY_LIB	= $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif
endif

ifdef HAVE_PYMOD
DATAFILES += $(PYMOD)

all: $(PYMOD)
endif

all: $(PYOBJS)

install: dir all
	$(RSYNC) $(PYSRCS) $(DATAFILES) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PIP) install -U $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel --plat-name $(BIN_PLATFORM)
	rm -rf $(WHL_BNDL_NAME).egg-info

setup.py: setup.py.in Makefile
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,SHLIB_DIR,$(shlibdir),' \
		-e 's,DATA_DIR,$(datadir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,#$(OS)#,,' \
		-e 's,#$(OSARCH)#,,' \
		-e 's,BUNDLE_NAME,$(BUNDLE_NAME),' \
		-e 's,BUNDLE_VERSION,$(BUNDLE_VERSION),' \
		-e 's,PKG_NAME,$(PKG_NAME),' \
		-e 's,PKG_DIR,$(PKG_DIR),' \
		< setup.py.in > setup.py

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ build dist $(WHL_BNDL_NAME).egg-info setup.py

ifdef HAVE_PYMOD

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
	# Fix _spacenavigator.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(PYMOD)
endif

endif
