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

MOUSEICONS = $(TOP)/src/bundles/mouse_modes/src/icons/*.png \
		$(TOP)/src/bundles/bond_rot/src/bondrot.png \
		$(TOP)/src/bundles/dist_ui/src/distance.png \
		$(TOP)/src/bundles/label/src/label.png \
		$(TOP)/src/bundles/label/src/movelabel.png \
		$(TOP)/src/bundles/map/src/contour.png \
		$(TOP)/src/bundles/map/src/moveplanes.png \
		$(TOP)/src/bundles/map/src/windowing.png \
		$(TOP)/src/bundles/map_series/src/vseries.png \
		$(TOP)/src/bundles/markers/src/icons/marker.png \
		$(TOP)/src/bundles/swapaa/src/swapaa.png \
		$(TOP)/src/bundles/tug/src/minimize.png \
		$(TOP)/src/bundles/tug/src/tug.png \
		$(TOP)/src/bundles/zone/src/zone.png
MICONDIR = $(DOCDIR)/mouse-icons
SHORTCUTICONS = $(TOP)/src/bundles/shortcuts/src/icons/*.png
SICONDIR = $(DOCDIR)/shortcut-icons
BASICICONS = $(TOP)/src/bundles/basic_actions/src/lib/*.svg
BICONDIR = $(DOCDIR)/basic-icons

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)
	if [ ! -d "$(MICONDIR)" ]; then mkdir -p $(MICONDIR); fi
	$(RSYNC) $(MOUSEICONS) $(MICONDIR)
	if [ ! -d "$(SICONDIR)" ]; then mkdir -p $(SICONDIR); fi
	$(RSYNC) $(SHORTCUTICONS) $(SICONDIR)
	if [ ! -d "$(BICONDIR)" ]; then mkdir -p $(SICONDIR); fi
	$(RSYNC) $(BASICICONS) $(BICONDIR)

clean:
