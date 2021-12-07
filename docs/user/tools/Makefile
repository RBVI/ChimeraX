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
		$(TOP)/src/bundles/color_key/src/icons/ckiconvertical.png \
		$(TOP)/src/bundles/label/src/movelabel.png \
		$(TOP)/src/bundles/map/src/icons/*.png \
		$(TOP)/src/bundles/map_series/src/vseries.png \
		$(TOP)/src/bundles/markers/src/icons/*.png \
		$(TOP)/src/bundles/std_commands/src/coordset.png \
		$(TOP)/src/bundles/swapaa/src/swapaa.png \
		$(TOP)/src/bundles/tug/src/minimize.png \
		$(TOP)/src/bundles/tug/src/tug.png \
		$(TOP)/src/bundles/zone/src/zone.png \
		$(TOP)/src/bundles/pick_blobs/src/pickblobs.png \
		$(TOP)/src/bundles/map_eraser/src/eraser.png \
		$(TOP)/src/bundles/tape_measure/src/tape.png \
		$(TOP)/src/bundles/viewdockx/src/nextdocking.png
MICONDIR = $(DOCDIR)/mouse-icons
SHORTCUTICONS = $(TOP)/src/bundles/shortcuts/src/icons/*.png
SICONDIR = $(DOCDIR)/shortcut-icons
TOOLBARICONS = $(TOP)/src/bundles/toolbar/src/icons/*.png \
		$(TOP)/src/bundles/medical_toolbar/src/icons/opendicom.png \
		$(TOP)/src/bundles/sel_inspector/src/icons/magnifying-glass.png
TICONDIR = $(DOCDIR)/toolbar-icons
NUCICONS = $(TOP)/src/bundles/nucleotides/src/icons/*.png
NICONDIR = $(DOCDIR)/nuc-icons
BASICICONS = $(TOP)/src/bundles/basic_actions/src/lib/*.svg
BICONDIR = $(DOCDIR)/basic-icons
UIICONS = $(TOP)/src/bundles/ui/src/icons/*.png
UICONDIR = $(DOCDIR)/ui-icons

install: install-icons
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

install-icons:
	if [ ! -d "$(MICONDIR)" ]; then mkdir -p $(MICONDIR); fi
	$(RSYNC) $(MOUSEICONS) $(MICONDIR)
	if [ ! -d "$(SICONDIR)" ]; then mkdir -p $(SICONDIR); fi
	$(RSYNC) $(SHORTCUTICONS) $(SICONDIR)
	if [ ! -d "$(TICONDIR)" ]; then mkdir -p $(TICONDIR); fi
	$(RSYNC) $(TOOLBARICONS) $(TICONDIR)
	if [ ! -d "$(NICONDIR)" ]; then mkdir -p $(NICONDIR); fi
	$(RSYNC) $(NUCICONS) $(NICONDIR)
	if [ ! -d "$(BICONDIR)" ]; then mkdir -p $(BICONDIR); fi
	$(RSYNC) $(BASICICONS) $(BICONDIR)
	if [ ! -d "$(UICONDIR)" ]; then mkdir -p $(UICONDIR); fi
	$(RSYNC) $(UIICONS) $(UICONDIR)

clean:
