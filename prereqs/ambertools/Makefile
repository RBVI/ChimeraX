PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using AmberTools distribution from https://ambermd.org

VERSION = 20

# to upload new binary versions, set USE_PREBUILT to false and UPLOAD_PREBUILT to true;
# once all platforms have been successfully built, set USE_PREBUILT to true and UPLOAD_PREBUILT to false
USE_PREBUILT=true
UPLOAD_PREBUILT=false
ifeq ($(OS),Linux)
# CentOS uses "CentOS Linux", hence the second awk in the following line
LINUX_FLAVOR = $(shell egrep '^NAME=' /etc/os-release | awk -F= '{ print $$2 }' | awk '{ print $$1 }' | tr -d '"')
LINUX_RELEASE = $(shell egrep '^VERSION_ID=' /etc/os-release | awk -F= '{ print $$2 }' | tr -d '"')
AMBERTOOLS_PREBUILT = ambertools-$(VERSION)-$(OS)-$(LINUX_FLAVOR)-$(LINUX_RELEASE).tar.bz2
else
AMBERTOOLS_PREBUILT = ambertools-$(VERSION)-$(OS).tar.bz2
ifeq ($(OS),Darwin)
AMBERTOOLS_FLIBS_PREBUILT = ambertools-flibs-$(VERSION)-$(OS).tar.bz2
FLIB_NAMES_FILE = $(tmpdir)/mac_flib_names
endif
endif

DISTRIBUTION = AmberTools$(VERSION).tar.bz2
SOURCE = $(tmpdir)/AmberTools$(VERSION)
AMBERHOME = $(SOURCE)/amber$(VERSION)_src
TOOL_SRC = $(AMBERHOME)/AmberTools/src
AMBERBIN_SRC = $(AMBERHOME)/bin
AMBERBIN_DST_HOME = $(app_bindir)/amber$(VERSION)
AMBERBIN_DST = $(AMBERBIN_DST_HOME)/bin
AMBERDAT_SRC = $(AMBERHOME)/dat/antechamber
AMBERDAT_DST = $(AMBERBIN_DST_HOME)/dat/antechamber

PATCHES	= $(wildcard AmberTools-*.patch)

test: $(SOURCE)

ENV_OPTS =
ifeq ($(OS),Windows)
CONFIG_ARGS = -cygwin gnu
ENV_OPTS = -u CC -u CXX FFLAGS=-fallow-argument-mismatch
PATCHES += flex-cygwin.patch
else
ifeq ($(OS),Darwin)
CONFIG_ARGS = clang
else
CONFIG_ARGS = gnu
endif
endif

ifeq ($(wildcard /usr/local/gfortran/*),)
FLIBDIR = /usr/local/lib
else
FLIBDIR = /usr/local/gfortran/lib
endif

ifeq ($(USE_PREBUILT),true)
install:

app-install: $(AMBERTOOLS_PREBUILT)
	-mkdir -p $(AMBERBIN_DST_HOME)
	 tar xf $(AMBERTOOLS_PREBUILT) -C $(AMBERBIN_DST_HOME)
ifeq ($(OS),Darwin)
	tar xf $(AMBERTOOLS_FLIBS_PREBUILT) -C $(app_libdir)
endif

$(AMBERTOOLS_PREBUILT):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ambertools/$(AMBERTOOLS_PREBUILT)
ifeq ($(OS),Darwin)
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ambertools/$(AMBERTOOLS_FLIBS_PREBUILT)
endif

else
install: $(SOURCE)
	cd $(AMBERHOME) && env $(ENV_OPTS) AMBERHOME='$(AMBERHOME)' ./configure --no-updates -noX11 -nofftw3 --skip-python $(CONFIG_ARGS)
	cd $(TOOL_SRC)/c9x-complex && env AMBERHOME='$(AMBERHOME)' $(MAKE) all
	cd $(TOOL_SRC)/cifparse && env AMBERHOME='$(AMBERHOME)' $(MAKE) install
	cd $(TOOL_SRC)/sqm && env AMBERHOME='$(AMBERHOME)' $(MAKE) FREEFORMAT_FLAG= install
	cd $(TOOL_SRC)/antechamber && env AMBERHOME='$(AMBERHOME)' $(MAKE) install

app-install: install
	-mkdir -p $(AMBERBIN_DST) $(AMBERDAT_DST)
	# Excluded files are because they reference netcdf and we do not need them
	$(RSYNC) $(AMBERBIN_SRC)/* --exclude nccopy --exclude ncdump --exclude ncgen --exclude ncgen3 --exclude to_be_dispatched $(AMBERBIN_DST)
	# Some executables live directly in bin, but many are shell scripts whose
	# executables are in the "to_be_dispatched" subdir; overwrite those...
	$(RSYNC) $(AMBERBIN_SRC)/to_be_dispatched/* --exclude nccopy --exclude ncdump --exclude ncgen --exclude ncgen3 $(AMBERBIN_DST)
ifeq ($(OS),Windows)
	$(RSYNC) /usr/bin/cygwin1.dll /usr/bin/cyggcc_s-seh*.dll /usr/bin/cyggfortran*.dll /usr/bin/cygquadmath*.dll /bin/sh.exe $(AMBERBIN_DST)
endif
	$(RSYNC) $(AMBERDAT_SRC)/* $(AMBERDAT_DST)
	#$(RSYNC) heme-iron.frcmod $(AMBERDAT_DST)/leap/parm
ifeq ($(OS),Darwin)
	# copy the shared libraries needed by the fortran executables and fix up their paths
	cat /dev/null > $(FLIB_NAMES_FILE)
	cd $(AMBERBIN_DST) && for i in sqm; do \
		for j in `otool -L $$i | awk '(NF > 1 && $$1 ~ /^$(subst /,\/,$(FLIBDIR))/) { print $$1 }'`; do \
			cp $$j $(app_libdir); \
			filename=`basename $$j`; \
			echo "$$filename " >> $(FLIB_NAMES_FILE); \
			install_filename=$(app_libdir)/$$filename; \
			install_name_tool -id @executable_path/$$filename $$install_filename; \
			install_name_tool -change $$j @executable_path/../../../lib/$$filename $$i; \
			for k in `otool -L $$j | awk '(NF > 1 && $$1 ~ /^$(subst /,\/,$(FLIBDIR))/) { print $$1 }'`; do \
				subfilename=`basename $$k`; \
				install_name_tool -change $$k @executable_path/../../../lib/$$subfilename $$install_filename; \
			done; \
		done; \
	done
endif
ifeq ($(UPLOAD_PREBUILT),true)
	tar cjf $(AMBERTOOLS_PREBUILT) -C "$(AMBERBIN_DST_HOME)" .
ifeq ($(OS),Darwin)
	tar cjf $(AMBERTOOLS_FLIBS_PREBUILT) -C "$(app_libdir)" `cat $(FLIB_NAMES_FILE)`
endif
	$(MAKE) upload-new-binary-version
endif
ifeq ($(OS),Darwin)
	rm $(tmpdir)/mac_flib_names
endif

$(SOURCE): $(DISTRIBUTION)
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(AMBERHOME) && patch -f -p0) < $$p ; \
	done
endif
# BUILD_TYPE is empty for local builds, otherwise one of daily, candidate, production
ifeq ($(BUILD_TYPE),daily)
	-cd $(AMBERHOME) && env AMBERHOME='$(AMBERHOME)' ./update_amber --update
else
	cd $(AMBERHOME) && env AMBERHOME='$(SOURCE)' ./update_amber --update
endif
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ambertools/$(DISTRIBUTION)

upload-new-source-version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/ambertools

upload-new-binary-version:
	$(RSYNC) $(AMBERTOOLS_PREBUILT) $(PREREQS_UPLOAD)/ambertools
ifeq ($(OS),Darwin)
	$(RSYNC) $(AMBERTOOLS_FLIBS_PREBUILT) $(PREREQS_UPLOAD)/ambertools
endif

clean:
	rm -rf $(SOURCE)
