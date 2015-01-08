TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/cmd_line
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/cmd_line

PYSRCS = $(wildcard *.py)
TOOL_VERSION = 1.0
WHEEL = dist/chimera.cmd_line-$(TOOL_VERSION)-py3-none-any.whl

all: $(PYOBJS)

install: dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	if [ -d "$(app_prefix)" ]; then \
		$(MAKE) app_install ;\
	fi

app_install: wheel
	$(APP_PYTHON_EXE) -m pip install --no-deps $(WHEEL)

wheel: setup.py always
	$(PYTHON_EXE) setup.py bdist_wheel

setup.py: setup.py.in
	sed -e 's,PKG_DIR,$(PKG_DIR),' \
		-e 's,LIB_DIR,$(libdir),' \
		-e 's,PYMOD_EXT,$(PYMOD_EXT),' \
		-e 's,SHLIB_EXT,$(SHLIB_EXT),' \
		-e 's,TOOL_VERSION,$(TOOL_VERSION),' \
		< setup.py.in > setup.py

always:

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

lint:
	$(PYLINT) $(PYSRCS)

test:
	$(bindir)/python3
clean:
	rm -rf __pycache__ build dist chimera.cmd_line.egg-info setup.py
