TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR	= $(PYSITEDIR)/chimera2/lighting
PYSRCS	= $(wildcard *.py)

all: $(PYOBJS)

install: all
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

lint:
	pyflakes $(PYSRCS)

clean:
	rm -rf __pycache__
