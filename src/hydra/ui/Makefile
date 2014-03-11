TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui

PYSRCS = __init__.py c2leap.py camera.py commands.py crossfade.py device.py gui.py \
	lightcmd.py modelpanel.py mousemodes.py qt.py shortcuts.py view.py

ICONS = icons

ifeq ($(OS),Darwin)
	SUBDIRS	= spacenavigator oculus
	include $(TOP)/mk/subdir.make
endif

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(ICONS) $(PKG_DIR)

$(SUBDIR_INSTALL): pkg_dir

clean:
	rm -rf __pycache__
