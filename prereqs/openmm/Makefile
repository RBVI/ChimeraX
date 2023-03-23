PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using OpenMM builds from https://anaconda.org/conda-forge/openmm/files
VERSION = 8.0.0

ifeq ($(OS),Darwin)
PLATFORM = mac
ifeq ($(MACOS_ARCH),arm64)
DISTRIBUTION = openmm-8.0.0-mac-py39ha14061b_0_apple.tar.bz2
else
DISTRIBUTION = openmm-8.0.0-mac-py39hc22aeb8_0_apple.tar.bz2
endif
endif
ifeq ($(OS),Linux)
PLATFORM = linux
# The Linux distribution comes from Tristan Croll
# compiled Oct 31, 2022 with the same compiler as ChimeraX.
# Using CUDA 11.2
DISTRIBUTION = openmm-8.0.0-linux-py39_cuda112.tar.bz2
endif
ifeq ($(OS),Windows)
PLATFORM = windows
# Using CUDA 11.2
DISTRIBUTION = openmm-8.0.0-win-py39h98f037e_0.tar.bz2
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
	  $(RSYNC) Lib/site-packages/openmm $(PYSITEDIR) ; \
	  $(RSYNC) Library/lib/ --exclude site-packages $(OPENMM_LIB_INSTALL)

else

install: $(SOURCE)
	cd $(SOURCE) ; \
	  $(RSYNC) lib/python$(PYTHON_VERSION)/site-packages/openmm $(PYSITEDIR) ; \
	  $(RSYNC) lib/ --exclude python$(PYTHON_VERSION) $(OPENMM_LIB_INSTALL)

endif

app-install: install
	$(RSYNC) $(OPENMM_LIB_INSTALL)/* $(app_shlibdir)
	$(RSYNC) $(PYSITEDIR)/openmm $(APP_PYSITEDIR)
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
	  chmod a+x site-packages/openmm/*.pyd ; \
	  chmod a+x site-packages/openmm/app/internal/*.pyd
	cd $(SOURCE)/Library/lib ; \
	  chmod a+x *dll *.lib plugins/*.dll plugins/*.lib
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openmm/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/openmm

clean:
	rm -rf $(SOURCE) $(DISTRIBUTION)
