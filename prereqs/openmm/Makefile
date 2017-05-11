PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using nightly OpenMM Mac and Linux builds from https://anaconda.org/omnia/openmm/files
PYVER = py36
ifeq ($(OS),Darwin)
VERSION = 7.1.1
PLATFORM = mac
endif
ifeq ($(OS),Linux)
VERSION = 7.1.1
PLATFORM = linux
endif
ifeq ($(OS),Windows)
VERSION = 7.0.1
PLATFORM = Windows
endif

DISTRIBUTION = openmm-$(VERSION)-$(PLATFORM)-$(PYVER).tar.bz2
SOURCE = $(tmpdir)/OpenMM-$(VERSION)-$(PLATFORM)
OPENMM_LIB_INSTALL = $(libdir)/openmm

PATCHES	= $(wildcard OpenMM-$(VERSION)-$(OS)-*.patch)
PATCHES += OpenMM-$(VERSION)-mmcif.patch

ifeq ($(OS),Windows)
# Forcefield patch from Tristan Croll for his ISOLDE tool.
# Allows simulating with missing bonds to adjoining residues.
PATCHES += OpenMM-$(VERSION)-forcefield.patch
endif

ifeq ($(OS),Windows)

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) python/simtk $(PYSITEDIR) ; \
	  $(RSYNC) lib $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) lib/python3.6/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) lib --exclude python3.6 $(OPENMM_LIB_INSTALL)

endif

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/lib/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)

$(SOURCE):
ifeq ($(OS),Windows)
	tar xf $(DISTRIBUTION) -C $(tmpdir)
else
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)
endif
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
