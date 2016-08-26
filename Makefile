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

TOP = .
TOP := $(shell (cd "$(TOP)"; pwd))
NO_SUBDIR_ALL=1
NO_SUBDIR_INSTALL=1
SUBDIRS = prereqs src

include $(TOP)/mk/config.make
include $(TOP)/mk/subdir.make

all:
	@echo "'make install' to build everything"

ifdef WIN32
install:	vsdefined
else
install:
endif
	@echo 'Started install at' `date` on `hostname`
	$(MAKE) build-dirs
	$(MAKE) -C prereqs install-prebuilt
	$(MAKE) -C prereqs app-install
	$(MAKE) -C src install
	$(MAKE) -C docs install
	@echo 'Finished install at' `date`

ifdef WIN32
vsdefined:
	if [ -z $${VSINSTALLDIR+x} ]; then \
		echo 'Visual Studio not found.  Run ". vsvars.sh"' ; \
		false; \
	fi
endif

docs.install:
	$(MAKE) -C docs install

build-dirs:
	-mkdir $(build_prefix) $(bindir) $(libdir) $(includedir) $(datadir) \
		$(webdir) $(wheelhouse)
ifndef WIN32
	-cd $(build_prefix) && ln -nfs lib lib64
endif
ifneq ($(libdir), $(shlibdir))
	-mkdir $(shlibdir)
endif
ifeq ($(OS),Darwin)
	-mkdir -p $(frameworkdir) $(build_prefix)/Library
	#-cd $(build_prefix)/Library && ln -nfs ../Frameworks .
endif

build-app-dirs:
	-mkdir -p $(app_prefix) $(app_bindir) $(app_libdir) $(app_datadir) \
		$(app_includedir) $(APP_PYSITEDIR)
ifeq ($(OS),Darwin)
	-mkdir -p $(app_prefix)/MacOS $(app_prefix)/Resources \
		$(app_frameworkdir)
endif

distclean: clean
	rm -rf $(build_prefix) $(app_prefix) prereqs/prebuilt-*.tar.bz2
	$(MAKE) -C prereqs/PyQt distclean

build-from-scratch:
	$(MAKE) distclean
	$(MAKE) install

# Linux debugging:

gltrace:
	rm -f $(APP_NAME).trace
	apitrace trace $(app_bindir)/$(APP_NAME) $(DATA_FILE)

dumptrace:
	@apitrace dump $(APP_NAME).trace
