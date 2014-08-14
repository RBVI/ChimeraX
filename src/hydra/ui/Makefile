TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui

PYSRCS = __init__.py c2leap.py cameracmd.py commands.py crossfade.py cyclecmd.py device.py filedialog.py gui.py \
	lightcmd.py materialcmd.py modelpanel.py motion.py mousemodes.py qt.py shortcuts.py silhouettecmd.py \
	turncmd.py view.py

ICONS = icons

ifeq ($(OS),Darwin)
	SUBDIRS	= spacenavigator oculus
	include $(TOP)/mk/subdir.make
endif

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(ICONS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
