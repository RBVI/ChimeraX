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

DISTRIBUTION = OpenMM-$(VERSION)-$(PLATFORM).tar.bz2
SOURCE = $(tmpdir)/OpenMM-$(VERSION)-$(PLATFORM)
OPENMM_LIB_INSTALL = $(libdir)/openmm

ifeq ($(OS),Windows)

install:

app-install:

else

install: $(SOURCE)
	cd $(SOURCE) ; printf "$(OPENMM_LIB_INSTALL)\n$(PYTHON_EXE)\n" | ./install.sh

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/lib/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)

$(SOURCE):
	tar xf $(DISTRIBUTION) -C $(tmpdir)

endif

clean:
	rm -rf $(SOURCE)
