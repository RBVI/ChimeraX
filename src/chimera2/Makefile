TOP = ../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera2

PYSRCS = $(wildcard *.py) io lighting
PYOBJS = $(PYSRCS:.py=.pyc)

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	$(RSYNC) ../shaders/*_webgl.txt ../shaders/*150.txt $(PKG_DIR)

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__

test_cmds:
	python3 cmds.py
