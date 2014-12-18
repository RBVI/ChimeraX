TOP = ../../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/map/data/dsn6

PYSRCS = __init__.py dsn6_format.py dsn6_grid.py writebrix.py

all: $(PYOBJS)

install: all
	-mkdir -p $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
