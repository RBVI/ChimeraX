TOP = ../../..
include $(TOP)/mk/config.make

BUNDLE_NAME = RBVI-IHM
# By convention, UCSF RBVI bundles start with "ChimeraX-"
BUNDLE_VERSION = 1.0
PKG_NAME = chimerax.ihm

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
WHEEL = dist/$(WHL_BNDL_NAME)-$(BUNDLE_VERSION)-py3-none-any.whl

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
	$(PYTHON_EXE) setup.py --no-user-cfg bdist_wheel
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
