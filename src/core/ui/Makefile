TOP= ../../..
include $(TOP)/mk/config.make

PYSRCS= $(wildcard *.py)
PKG_DIR= $(PYSITEDIR)/chimera/core/ui
DATA = splash.jpg

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

lint:
	python3 -m flake8 $(PYSRCS)

clean:
	rm -rf __pycache__
