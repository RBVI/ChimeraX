PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.3.1
EGG_VERSION = 1.3
DISTRIBUTION = pdbfixer-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/pdbfixer-$(VERSION)

PATCHES	= pdbfixer-$(VERSION)-openmm.patch pdbfixer-$(VERSION)-mmcif.patch

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install

app-install: install
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py install

$(SOURCE):
	tar xf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
