PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using OpenMM builds from https://anaconda.org/omnia/openmm/files

VERSION = 7.5.0
OPENMM_REL = 1
PYVER = py38

ifeq ($(OS),Darwin)
PLATFORM = mac
CUDA_VERSION = nocuda
endif
ifeq ($(OS),Linux)
PLATFORM = linux
CUDA_VERSION = cuda110
endif
ifeq ($(OS),Windows)
PLATFORM = windows
CUDA_VERSION = cuda110
endif

DISTRIBUTION = openmm-$(VERSION)-$(PLATFORM)-$(PYVER)_$(CUDA_VERSION)_$(OPENMM_REL).tar.bz2
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
	  $(RSYNC) Library/lib/ --exclude site-packages $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) lib/python$(PYTHON_VERSION)/site-packages/simtk $(PYSITEDIR) ; \
	  $(RSYNC) lib/ --exclude python$(PYTHON_VERSION) $(OPENMM_LIB_INSTALL)

endif

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/simtk $(APP_PYSITEDIR)
ifeq ($(OS),Windows)
	cd $(app_shlibdir); mv OpenMM*.lib libs/
endif

$(SOURCE): $(DISTRIBUTION)
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
ifeq ($(OS),Windows)
	cd $(SOURCE)/Lib ; \
	  chmod a+x site-packages/simtk/openmm/*.pyd ; \
	  chmod a+x site-packages/simtk/openmm/app/internal/*.pyd
	cd $(SOURCE)/Library/lib ; \
	  chmod a+x *dll *.lib plugins/*.dll plugins/*.lib
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openmm/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/openmm

clean:
	rm -rf $(SOURCE)
