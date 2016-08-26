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

TOP = ..

include $(TOP)/mk/config.make

ifdef WIN32
RST2HTML = $(shell cygpath -m $(bindir)/Scripts/rst2html.py)
else
RST2HTML = $(bindir)/rst2html.py
endif

SUBDIRS = devel pdbimages quickstart user download

DOCDIR = $(app_datadir)/docs

SRCS = $(wildcard *.html) license.html
IMAGES = $(wildcard *.png)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install: license.html
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

license.html: license.rst
	$(PYTHON_EXE) $(RST2HTML) \
		--smart-quotes=alt \
		--stylesheet-path=license.css \
		license.rst > license.html

clean:
	rm -f license.html
