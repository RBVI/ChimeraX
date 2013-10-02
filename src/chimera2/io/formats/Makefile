TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR	= $(PYSITEDIR)/chimera2/io/formats

PYSRCS	= $(wildcard *.py)

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__
