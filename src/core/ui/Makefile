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

TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS	+= widgets

include $(TOP)/mk/subdir.make

PYSRCS= $(wildcard *.py)
PKG_DIR= $(PYSITEDIR)/chimerax/core/ui
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/ui
DATA = splash.jpg contract*.png expand*.png lightning*.png about.html

all: $(PYSRCS)
	$(PYTHON_EXE) -B -m py_compile $?

install: all dir about.html
	$(RSYNC) $(PYSRCS) $(DATA) $(PKG_DIR)
	#if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

clean:
	rm -rf __pycache__

lint:
	$(PYLINT) $(PYSRCS)

about.html:	about.html.in Makefile
	sed -e '/===/d' \
		-e 's,http:.*html,<a href="&">&</a>.,' \
		$(TOP)/copyright.txt | \
		sed $$'/COPYRIGHT_NOTICE/{r /dev/stdin\nd\n}' \
		about.html.in > about.html \
		|| rm about.html
