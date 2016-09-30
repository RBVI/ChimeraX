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

# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= core bundles apps wheelhouse wsgi

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all:

mainapp-install: core.install
	$(MAKE) -C apps/$(APP_NAME) install

core.install: wheelhouse.install
bundles.install: core.install mainapp-install
apps.install: bundles.install core.install
