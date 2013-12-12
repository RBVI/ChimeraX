TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui

PYSRCS = c2leap.py camera.py commands.py crossfade.py gui.py mousemodes.py qt.py shortcuts.py view.py

ICONS = icons

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(ICONS) $(PKG_DIR)

clean:
	rm -rf __pycache__
