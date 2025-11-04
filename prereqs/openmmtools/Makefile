PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.25.1
MPIPLUS_VERSION = 0+untagged.38.gfd8a5aa.dirty

DISTRIBUTION = openmmtools-$(VERSION).tar.bz2
MPIPLUS_DISTRIBUTION = mpiplus-$(MPIPLUS_VERSION).tar.bz2
OPENMM_LIB_INSTALL = $(libdir)/openmm

install: mpiplus-install $(DISTRIBUTION)
	tar xzf $(DISTRIBUTION) -C $(PYSITEDIR)

app-install: mpiplus-app-install install
	$(RSYNC) $(PYSITEDIR)/openmmtools $(APP_PYSITEDIR)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openmmtools/$(DISTRIBUTION)

mpiplus-install: $(MPIPLUS_DISTRIBUTION)
	tar xzf $(MPIPLUS_DISTRIBUTION) -C $(PYSITEDIR)

mpiplus-app-install: mpiplus-install
	$(RSYNC) $(PYSITEDIR)/mpiplus $(APP_PYSITEDIR)

$(MPIPLUS_DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/mpiplus/$(MPIPLUS_DISTRIBUTION)

clean:
	rm -rf $(SOURCE)
