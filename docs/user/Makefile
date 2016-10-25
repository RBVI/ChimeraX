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
TOP	= ../..
include $(TOP)/mk/config.make

SUBDIRS = tools

DOCDIR = $(app_datadir)/docs/user

SRCS = $(wildcard *.html) userdocs.css commands
IMAGES = $(wildcard *.png) $(wildcard *.svg)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

clean:
