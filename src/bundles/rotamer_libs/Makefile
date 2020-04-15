SUBDIRS = Dunbrack Dynameomics Richardson
include ../../../mk/subdir.make

SUBDIR_SYNC = $(SUBDIRS:=.sync)
.PHONY: $(SUBDIR_SYNC)
sync: $(SUBDIR_SYNC)
$(SUBDIR_SYNC):
	$(MAKE) -C $(subst .install,,$@) sync
