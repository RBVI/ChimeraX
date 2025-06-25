# -*- mode: make -*- vim: set syntax=make:
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

# Define SUBDIRS with a list of subdirectories to make
# before including this file.  The convolutions below
# enable parallel makes across subdirectories.
#
# The targets are all, install, clean, and test.

ifdef NO_SUBDIR_INSTALL
install:
else
SUBDIR_INSTALL = $(SUBDIRS:=.install)
.PHONY: $(SUBDIR_INSTALL)
install: $(SUBDIR_INSTALL)
$(SUBDIR_INSTALL):
	$(MAKE) -C $(subst .install,,$@) install
endif

ifdef NO_SUBDIR_CLEAN
clean:
else
SUBDIR_CLEAN = $(SUBDIRS:=.clean)
.PHONY: $(SUBDIR_CLEAN)
clean: $(SUBDIR_CLEAN)
$(SUBDIR_CLEAN):
	$(MAKE) -C $(subst .clean,,$@) clean
endif

ifdef NO_SUBDIR_TEST
test:
else
SUBDIR_TEST = $(SUBDIRS:=.test)
.PHONY: $(SUBDIR_TEST)
test: $(SUBDIR_TEST)
$(SUBDIR_TEST):
	$(MAKE) -C $(subst .test,,$@) test
endif

ifdef SUBDIR_SYNC
SUBDIR_SYNC = $(SUBDIRS:=.sync)
.PHONY: $(SUBDIR_SYNC)
sync: $(SUBDIR_SYNC)
$(SUBDIR_SYNC):
	$(MAKE) -C $(subst .sync,,$@) sync
else
sync:
endif

ifdef SUBDIR_BUILD
SUBDIR_BUILD = $(SUBDIRS:=.build)
UV_SUBDIR_BUILD = $(SUBDIRS:=.uv-build)
.PHONY: $(SUBDIR_BUILD) $(UV_SUBDIR_BUILD)

build: $(SUBDIR_BUILD)
$(SUBDIR_BUILD):
	$(MAKE) -C $(subst .build,,$@) build

uv-build: $(UV_SUBDIR_BUILD)
$(UV_SUBDIR_BUILD):
	-PYTHON=python $(MAKE) -C $(subst .uv-build,,$@) uv-prepare
	cd $(subst .uv-build,,$@) && if [ -e bundle_info.xml ]; then python -m chimerax.bundle_builder -c bundle_info.xml > pyproject.toml; fi
	PYTHON=python $(MAKE) -C $(subst .uv-build,,$@) uv-build
	cd $(subst .uv-build,,$@) && if [ -e bundle_info.xml ]; then rm pyproject.toml; fi
else
build:
uv-build:
endif
