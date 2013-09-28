TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui

PYSRCS = c2leap.py commands.py crossfade.py gui.py qt.py shortcuts.py view.py

ICONS = icons

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(ICONS) $(PKG_DIR)

clean:
	rm -rf __pycache__
