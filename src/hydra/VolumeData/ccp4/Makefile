TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/VolumeData/ccp4

PYSRCS = __init__.py ccp4_grid.py

all: $(PYOBJS)

install: all
	-mkdir -p $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
