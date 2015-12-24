TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/commands
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/commands

GENSRCS = _atomspec.py
PYSRCS = $(wildcard *.py) $(GENSRCS)

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

.SUFFIXES: .peg

.peg.py:
	$(PYTHON_EXE) -m grako -o $@ $^

test_peg:	_atomspec.py
	$(PYTHON_EXE) _atomspec.py --trace atomspec_test.in atom_specifier

clean:
	rm -rf __pycache__ $(GENSRCS)

lint:
	$(PYLINT) $(PYSRCS)
