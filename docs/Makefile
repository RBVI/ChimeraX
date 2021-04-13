# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = ..

include $(TOP)/mk/config.make

ifdef WIN32
RST2HTML = $(shell cygpath -m $(app_bindir)/Scripts/rst2html.py)
HTML2TEXT = $(app_bindir)/Scripts/html2text.exe
else
RST2HTML = $(app_bindir)/rst2html.py
endif

SUBDIRS = devel pdbimages quickstart user js

DOCDIR = $(app_datadir)/docs

ifndef UBUNTU_DIST
EXTRA_EMBEDDED = ffmpeg
endif

SRCS = $(wildcard *.html) license.html embedded.html licenses videos
IMAGES = $(wildcard *.png)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install: license.html embedded.html need_app
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

$(SUBDIR_INSTALL): need_app

need_app:
	@if [ ! -d "$(app_prefix)" ]; then \
		echo "!! Need application to install documentation !!"; \
		exit 1; \
	fi

# rst2html converts quotes to Unicode.
# We turn them back into HTML entities.
license.html: license.rst need_app
	$(APP_PYTHON_EXE) $(RST2HTML) \
			--no-xml-declaration \
			--smart-quotes=alt \
			--stylesheet-path=license.css \
			license.rst \
		| sed -e 's/\o342\o200\o234/\&ldquo;/g' \
			-e 's/\o342\o200\o235/\&rdquo;/g' \
			-e "s/\o342\o200\o231/'/g" \
		> license.html
ifdef WIN32
	$(APP_PYTHON_EXE) -m pip install html2text
	$(HTML2TEXT) --body-width=0 license.html > license.txt
endif

embedded.html licenses: embedded.html.in make_embedded.py need_app
	if [ -d "$(app_prefix)" ]; then \
		$(APP_PYTHON_EXE) make_embedded.py $(EXTRA_EMBEDDED); \
	fi
	cp -p basictypes-license.txt licenses


clean:
	rm -f license.html license.txt embedded.html
	rm -rf licenses
