TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/file_io/SessionUtil

PYSRCS = __init__.py objecttree.py stateclasses.py

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
