TOP = ../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera2

PYSRCS = $(wildcard *.py)
PYOBJS = $(PYSRCS:.py=.pyc)

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf $(PYOBJS)
