# Define SUBDIRS with a list of subdirectories to make
# before including this file.  The convolutions below
# enable parallel makes across subdirectories.
#
# The targets are the same as common.make:
# all, install, clean, and distclean.

ifeq ($(MAKECMDGOALS),distclean)
NO_SUBDIR_CLEAN = true
endif

ifdef NO_SUBDIR_ALL
all:
else
SUBDIR_ALL = $(SUBDIRS:=.all)
.PHONY: $(SUBDIR_ALL)
all: $(SUBDIR_ALL)
$(SUBDIR_ALL):
	$(MAKE) -C $(subst .all,,$@)
endif

ifdef NO_SUBDIR_INSTALL
install:
else
SUBDIR_INSTALL = $(SUBDIRS:=.install)
.PHONY: $(SUBDIR_INSTALL)
install: $(SUBDIR_INSTALL)
ifndef NO_SUBDIR_ALL
$(SUBDIR_INSTALL): %.install: %.all
else
$(SUBDIR_INSTALL):
endif
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

ifdef NO_SUBDIR_DISTCLEAN
distclean:
else
SUBDIR_DISTCLEAN = $(SUBDIRS:=.distclean)
.PHONY: $(SUBDIR_DISTCLEAN)
distclean: $(SUBDIR_DISTCLEAN)
$(SUBDIR_DISTCLEAN):
	$(MAKE) -C $(subst .distclean,,$@) distclean
endif
