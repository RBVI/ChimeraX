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

# Makefile for Javascript files.
#
TOP	= ../..
include $(TOP)/mk/config.make

JSDIR = $(app_datadir)/docs/js

JS_SRCS = cxlinks.js

install:
	if [ ! -d "$(JSDIR)" ]; then mkdir -p $(JSDIR); fi
	$(RSYNC) $(JS_SRCS) $(JSDIR)

clean:
