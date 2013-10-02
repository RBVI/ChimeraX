TOP	= ../../..
include $(TOP)/mk/config.make

SUBDIRS	= formats
include $(TOP)/mk/subdir.make

PKG_DIR	= $(PYSITEDIR)/chimera2/io
PYSRCS	= $(wildcard *.py)

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__
