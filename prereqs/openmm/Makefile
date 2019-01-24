PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using nightly OpenMM Mac and Linux builds from https://anaconda.org/omnia/openmm/files
PYVER = py37_cuda92_rc_1
VERSION = 7.3.0
ifeq ($(OS),Darwin)
PLATFORM = mac
endif
ifeq ($(OS),Linux)
PLATFORM = linux
endif
ifeq ($(OS),Windows)
PYVER = py37_cuda92_rch39e3cac_1
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
PATCHES	= $(wildcard OpenMM-$(OS)-*.patch)

ifeq ($(OS),Windows)

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) Lib/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) Lib/ --exclude site-packages $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) lib/python$(PYTHON_VERSION)/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) lib/ --exclude python$(PYTHON_VERSION) $(OPENMM_LIB_INSTALL)

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
ifeq ($(OS),Windows)
	cd $(SOURCE)/Lib ; \
	  chmod a+x *dll *.lib plugins/*.dll plugins/*.lib ; \
	  chmod a+x site-packages/simtk/openmm/_openmm*.pyd
endif

clean:
	rm -rf $(SOURCE)
