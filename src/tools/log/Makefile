TOP = ../../..
include $(TOP)/mk/config.make

#TOOL = FULL_PACAKGE_NAME
TOOL = chimera.log
TOOL_VERSION = 1.0

TOOL_PATH = $(shell echo $(TOOL) | sed 's,/,.,')
PKG_DIR = $(PYSITEDIR)/$(TOOL_PATH)
APP_PKG_DIR = $(APP_PYSITEDIR)/$(TOOL_PATH)

PYSRCS = $(wildcard *.py)
WHEEL = dist/$(TOOL)-$(TOOL_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app-install ;\
	fi

app-install: wheel
	$(APP_PYTHON_EXE) -m pip install --no-deps $(WHEEL)
	$(RSYNC) $(WHEEL) $(wheelhouse) && cd $(wheelhouse) && make update

wheel: setup.py timestamp
	$(PYTHON_EXE) setup.py bdist_wheel

setup.py: setup.py.in
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,TOOL_VERSION,$(TOOL_VERSION),' \
		-e 's,TOOL,$(TOOL),' \
		< setup.py.in > setup.py

timestamp: always
	echo "$(TOOL) $(TOOL_VERSION)" >  timestamp

always:

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ build dist $(TOOL).egg-info setup.py timestamp
