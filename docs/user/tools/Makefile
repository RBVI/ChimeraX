# Makefile for user guide
#
TOP	= ../../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/user/tools

SRCS = $(wildcard *.html)

MOUSEICONS = $(TOP)/src/tools/mouse_modes/icons/*.png
MICONDIR = $(DOCDIR)/mouse-icons
SHORTCUTICONS = $(TOP)/src/tools/shortcuts/icons/*.png
SICONDIR = $(DOCDIR)/shortcut-icons

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(DOCDIR)
	if [ ! -d "$(MICONDIR)" ]; then mkdir -p $(MICONDIR); fi
	$(RSYNC) $(MOUSEICONS) $(MICONDIR)
	if [ ! -d "$(SICONDIR)" ]; then mkdir -p $(SICONDIR); fi
	$(RSYNC) $(SHORTCUTICONS) $(SICONDIR)

clean:
