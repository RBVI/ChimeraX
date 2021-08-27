# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Makefile for CSS files.
#
TOP	= ../..
include $(TOP)/mk/config.make

CSSDIR = $(app_datadir)/docs/css

CSS_SRCS = devdocs.css

install:
	if [ ! -d "$(CSSDIR)" ]; then mkdir -p $(CSSDIR); fi
	$(RSYNC) $(CSS_SRCS) $(CSSDIR)

clean:
