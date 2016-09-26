# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Makefile for user guide
#
TOP	= ../../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/user/tools

SRCS = $(wildcard *.html)
IMAGES = $(wildcard *.png)

MOUSEICONS = $(TOP)/src/bundles/mouse_modes/icons/*.png
MICONDIR = $(DOCDIR)/mouse-icons
SHORTCUTICONS = $(TOP)/src/bundles/shortcuts/icons/*.png
SICONDIR = $(DOCDIR)/shortcut-icons

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)
	if [ ! -d "$(MICONDIR)" ]; then mkdir -p $(MICONDIR); fi
	$(RSYNC) $(MOUSEICONS) $(MICONDIR)
	if [ ! -d "$(SICONDIR)" ]; then mkdir -p $(SICONDIR); fi
	$(RSYNC) $(SHORTCUTICONS) $(SICONDIR)

clean:
