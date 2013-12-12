TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/surface

PYSRCS = __init__.py gridsurf.py sasa.py shapes.py split.py surface.py tube.py

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
