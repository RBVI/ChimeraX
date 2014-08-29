TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/ui

SUBDIRS = qt wx

PYSRCS = __init__.py log.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

ICONS = icons

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(ICONS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
