PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using OpenMM builds from https://anaconda.org/omnia/openmm/files

VERSION = 7.5.0

ifeq ($(OS),Darwin)
PLATFORM = mac
DISTRIBUTION = openmm-7.5.0-mac-py38_nocuda_1.tar.bz2
#DISTRIBUTION = openmm-7.5.1-mac-py39h17f8b25_1_apple.tar.bz2
endif
ifeq ($(OS),Linux)
PLATFORM = linux
DISTRIBUTION = openmm-7.5.0-linux-py38_cuda110_1.tar.bz2
endif
ifeq ($(OS),Windows)
PLATFORM = windows
DISTRIBUTION = openmm-7.5.0-windows-py38_cuda110_1.tar.bz2
endif

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
