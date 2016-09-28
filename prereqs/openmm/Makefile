PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 7.0.1
ifeq ($(OS),Darwin)
PLATFORM = Mac
endif
ifeq ($(OS),Linux)
PLATFORM = Linux
endif
ifeq ($(OS),Windows)
PLATFORM = Windows
endif

DISTRIBUTION = OpenMM-$(VERSION)-$(PLATFORM).tar.bz2
SOURCE = $(tmpdir)/OpenMM-$(VERSION)-$(PLATFORM)
OPENMM_LIB_INSTALL = $(libdir)/openmm

PATCHES	= $(wildcard OpenMM-$(VERSION)-$(PLATFORM)-*.patch)

ifeq ($(OS),Windows)

install: $(SOURCE)
	cd $(SOURCE) ; $(RSYNC) simtk $(PYSITEDIR) ; $(RSYNC) lib $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; printf "$(OPENMM_LIB_INSTALL)\n$(PYTHON_EXE)\n" | ./install.sh

endif

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/lib/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)

$(SOURCE):
	tar xf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
