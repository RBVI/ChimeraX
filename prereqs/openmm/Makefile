PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using nightly OpenMM Mac and Linux builds from https://anaconda.org/omnia/openmm/files
PYVER = py36
VERSION = 7.1.1
ifeq ($(OS),Darwin)
PLATFORM = mac
endif
ifeq ($(OS),Linux)
PLATFORM = linux
endif
ifeq ($(OS),Windows)
PLATFORM = windows
endif

DISTRIBUTION = openmm-$(VERSION)-$(PLATFORM)-$(PYVER).tar.bz2
SOURCE = $(tmpdir)/OpenMM-$(VERSION)-$(PLATFORM)
OPENMM_LIB_INSTALL = $(libdir)/openmm

# Beware the Windows OpenMM has the Python code in a different location
# than Mac and Linux, so one patch for OpenMM Python code won't find the
# Python file on all 3 platforms.  Also the Windows Python has /r/n DOS
# line endings and patches expecting /n line endings will fail.
# Probably easiest to make a separate Windows patch.
PATCHES	= $(wildcard OpenMM-$(VERSION)-$(OS)-*.patch)

ifeq ($(OS),Windows)

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) Lib/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) Lib/ --exclude site-packages $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) lib/python3.6/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) lib/ --exclude python3.6 $(OPENMM_LIB_INSTALL)

endif

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)

$(SOURCE):
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
