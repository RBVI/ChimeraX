TOP = ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/map/data/mrc

PYSRCS = __init__.py mrc_format.py mrc_grid.py writemrc.py

all: $(PYOBJS)

install: all
	-mkdir -p $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
