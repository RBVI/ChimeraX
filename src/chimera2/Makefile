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
	-mkdir $(PKG_DIR)

$(SUBDIR_INSTALL): dir

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__

test_cmds:
	python3 cmds.py
