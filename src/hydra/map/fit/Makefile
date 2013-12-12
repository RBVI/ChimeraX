TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/map/fit

PYSRCS = __init__.py fitcmd.py fitlist.py fitmap.py move.py \
	search.py sequence.py session.py

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
