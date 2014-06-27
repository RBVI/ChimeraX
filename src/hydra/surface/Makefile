TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/surface

PYSRCS = __init__.py gridsurf.py sasa.py shapes.py split.py tube.py

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
