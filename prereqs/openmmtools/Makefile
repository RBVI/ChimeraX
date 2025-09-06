PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.25.1

DISTRIBUTION = openmmtools-$(VERSION).tar.bz2
OPENMM_LIB_INSTALL = $(libdir)/openmm

install: $(DISTRIBUTION)
	tar xzf $(DISTRIBUTION) -C $(PYSITEDIR)

app-install: install
	$(RSYNC) $(PYSITEDIR)/openmmtools $(APP_PYSITEDIR)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openmmtools/$(DISTRIBUTION)

clean:
	rm -rf $(SOURCE)
