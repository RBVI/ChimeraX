# Define SUBDIRS with a list of subdirectories to make
# before including this file.  The convolutions below
# enable parallel makes across subdirectories.
#
# The targets are all, install, and clean.

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
