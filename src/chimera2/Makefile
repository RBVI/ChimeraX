TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS = io lighting

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimera2

PYSRCS = $(wildcard *.py)

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	$(RSYNC) ../shaders/*_webgl.txt ../shaders/*150.txt $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
	$(PYTHON_EXE) -m flake8 $(PYSRCS)

clean:
	rm -rf __pycache__

test_cmds:
	$(PYTHON_EXE) cmds.py
