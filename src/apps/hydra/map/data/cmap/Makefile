TOP = ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/map/data/cmap

PYSRCS = __init__.py cmap_format.py cmap_grid.py write_cmap.py

all: $(PYOBJS)

install: all
	-mkdir -p $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
