PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using AmberTools distribution from https://ambermd.org

#ifeq ($(OS),Darwin)
#VERSION = 24
#SRC_VERSION = 24
#else
VERSION = 20
SRC_VERSION = 20
#endif

# to upload new binary versions, set USE_PREBUILT to false and UPLOAD_PREBUILT to true;
# once all platforms have been successfully built, set USE_PREBUILT to true and UPLOAD_PREBUILT to false
#ifeq ($(OS),Darwin)
#USE_PREBUILT=false
#UPLOAD_PREBUILT=true
#else
USE_PREBUILT=true
UPLOAD_PREBUILT=false
#endif
ifeq ($(OS),Linux)
SUPPORTED_DISTROS := Ubuntu CentOS Debian Rocky Freedesktop
# CentOS uses "CentOS Linux", hence the second awk in the following line
LINUX_FLAVOR = $(shell grep '^NAME=' /etc/os-release | sed -e 's/.*="//' -e 's/\( .*\)*"//')
LINUX_RELEASE = $(shell grep '^VERSION_ID=' /etc/os-release | sed -e 's/.*="*//' -e 's/\..*//')
AMBERTOOLS_PREBUILT = ambertools-$(VERSION)-$(OS)-$(LINUX_FLAVOR)-$(LINUX_RELEASE).tar.bz2
ifdef FLATPAK_DIST
ifeq ($(OSARCH),LinuxArm64)
AMBERTOOLS_PREBUILT = ambertools-24-$(OS)-arm-$(LINUX_FLAVOR)-$(LINUX_RELEASE).tar.bz2
else
AMBERTOOLS_PREBUILT = ambertools-24-$(OS)-$(LINUX_FLAVOR)-$(LINUX_RELEASE).tar.bz2
endif
endif
ifneq ($(filter $(LINUX_FLAVOR),$(SUPPORTED_DISTROS)),)
else
USE_PREBUILT=false
endif
else
AMBERTOOLS_PREBUILT = ambertools-$(VERSION)-$(OS).tar.bz2
ifeq ($(OS),Darwin)
AMBERTOOLS_FLIBS_PREBUILT = ambertools-flibs-$(VERSION)-$(OS).tar.bz2
FLIB_NAMES_FILE = $(tmpdir)/mac_flib_names
endif
endif

CMAKE = cmake

DISTRIBUTION = AmberTools$(VERSION).tar.bz2
SOURCE = $(tmpdir)/AmberTools$(VERSION)
AMBERHOME = $(SOURCE)/amber$(SRC_VERSION)_src
CMAKE_BUILD_DIR = $(SOURCE)/amber$(SRC_VERSION)_build
CMAKE_INSTALL_DIR = $(SOURCE)/amber$(SRC_VERSION)_install
CMAKE_TOOL_SRC = $(CMAKE_BUILD_DIR)/AmberTools/src
TOOL_SRC = $(AMBERHOME)/AmberTools/src
AMBERBIN_SRC = $(CMAKE_INSTALL_DIR)/bin
AMBERLIB_SRC = $(CMAKE_INSTALL_DIR)/lib
AMBERBIN_DST_HOME = $(app_bindir)/amber$(VERSION)
AMBERBIN_DST = $(AMBERBIN_DST_HOME)/bin
AMBERDAT_SRC = $(AMBERHOME)/dat
AMBERDAT_DST = $(AMBERBIN_DST_HOME)/dat
AMBERDAT_ANTECHAMBER_SRC = $(AMBERDAT_SRC)/antechamber
AMBERDAT_ANTECHAMBER_DST = $(AMBERDAT_DST)/antechamber
AMBERDAT_PARM_SRC = $(AMBERDAT_SRC)/leap/parm
AMBERDAT_PARM_DST = $(AMBERDAT_DST)/leap/parm

PATCHES	= $(wildcard AmberTools-*.patch)

test: $(SOURCE)

ENV_OPTS =
ifeq ($(OS),Windows)
CONFIG_ARGS = -cygwin gnu
ENV_OPTS = -u CC -u CXX FFLAGS=-fallow-argument-mismatch
PATCHES += flex-cygwin.patch
else ifeq ($(OS),Linux)
CONFIG_ARGS = gnu
ENV_OPTS = FFLAGS=-fallow-argument-mismatch
ifdef UBUNTU_DIST
ifeq ($(shell expr $(UBUNTU_DIST) '>' 18.04),1)
ENV_OPTS += CC=gcc-9 FC=gfortran-9
endif
endif
else ifeq ($(OS),Darwin)
CONFIG_ARGS = clang
endif

LIBGCCDIR=
ifeq ($(wildcard /usr/local/gfortran/*),)
ifeq ($(wildcard /usr/local/opt/gcc/lib/gcc/current/*),)
ifeq ($(wildcard /opt/homebrew/opt/gcc/lib/gcc/current/*),)
FLIBDIR = /usr/local/lib
else
FLIBDIR = /opt/homebrew/opt/gcc/lib/gcc/current
LIBGCCDIR = /opt/homebrew/opt/gcc/lib/gcc/current
endif
else
FLIBDIR = /usr/local/opt/gcc/lib/gcc/current
endif
else
FLIBDIR = /usr/local/gfortran/lib
endif

ifeq ($(USE_PREBUILT),true)
install:

app-install: $(AMBERTOOLS_PREBUILT)
	-mkdir -p $(AMBERBIN_DST_HOME)
ifeq ($(OS),Windows)
	# Antechamber complains bitterly about a missing tmp directory otherwise
	-mkdir -p $(AMBERBIN_DST_HOME)/tmp
endif
	 tar xf $(AMBERTOOLS_PREBUILT) -C $(AMBERBIN_DST_HOME) --no-same-owner
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
	-mkdir $(CMAKE_BUILD_DIR) $(CMAKE_INSTALL_DIR)
	cd $(CMAKE_BUILD_DIR) && env PATH=$(PATH):/opt/homebrew/bin:/usr/local/bin $(CMAKE) $(AMBERHOME) -DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_DIR) -DCOMPILER=CLANG -DDOWNLOAD_MINICONDA=FALSE -DBUILD_PYTHON=FALSE -DFORCE_INTERNAL_LIBS=netcdf
	for i in netcdf-4.6.1 netcdf-fortran-4.4.4 arpack sff cifparse ; do \
		if [ -e $(CMAKE_TOOL_SRC)/$$i ] ; then \
			cd $(CMAKE_TOOL_SRC)/$$i && $(MAKE) install ; \
		fi ; \
	done
	cd $(CMAKE_TOOL_SRC)/sqm && $(MAKE) install
	cd $(CMAKE_TOOL_SRC)/antechamber && $(MAKE) install

app-install: install
	-mkdir -p $(AMBERBIN_DST) $(AMBERDAT_ANTECHAMBER_DST) $(AMBERDAT_PARM_DST)
	$(RSYNC) $(AMBERBIN_SRC)/* --exclude wrapped_progs $(AMBERBIN_DST)
	# Some executables live directly in bin, but many are shell scripts whose
	# executables are in the "wrapped_progs" subdir; overwrite those...
	$(RSYNC) $(AMBERBIN_SRC)/wrapped_progs/* $(AMBERBIN_DST)
	$(RSYNC) $(AMBERDAT_ANTECHAMBER_SRC)/* $(AMBERDAT_ANTECHAMBER_DST)
	$(RSYNC) $(AMBERDAT_PARM_SRC)/gaff2.dat $(AMBERDAT_PARM_DST)
	$(RSYNC) $(AMBERLIB_SRC)/*.$(SHLIB_EXT) $(app_libdir)
ifneq ($(LIBGCCDIR),)
	for gcclib in $(LIBGCCDIR)/libgcc*.$(SHLIB_EXT) ; do \
		$(RSYNC) $$gcclib $(app_libdir); \
		filename=`basename $$gcclib`; \
		install_name_tool -id @executable_path/$$filename $(app_libdir)/$$filename; \
	done;
endif
ifeq ($(OS),Windows)
	$(RSYNC) /usr/bin/cygwin1.dll /usr/bin/cyggcc_s-seh*.dll /usr/bin/cyggfortran*.dll /usr/bin/cygquadmath*.dll /usr/bin/cygreadline*.dll /usr/bin/cygintl*.dll /usr/bin/cygncursesw*.dll /usr/bin/cygiconv*.dll /bin/sh.exe $(AMBERBIN_DST)
endif
ifeq ($(OS),Darwin)
	install_name_tool -rpath @loader_path/../../lib @loader_path/../../../lib $(AMBERBIN_DST)/antechamber
	install_name_tool -rpath @loader_path/../lib @loader_path/../../../lib $(AMBERBIN_DST)/sqm
	# copy the shared libraries needed by the fortran executables and fix up their paths
	# then do the same for libs that those shared libraries use
	cat /dev/null > $(FLIB_NAMES_FILE)
	cd $(AMBERBIN_DST) && for i in sqm; do \
		for j in `otool -L $$i | awk '(NF > 1 && $$1 ~ /^$(subst /,\/,$(FLIBDIR))/) { print $$1 }'`; do \
			cp $$j $(app_libdir); \
			filename=`basename $$j`; \
			echo "$$filename " >> $(FLIB_NAMES_FILE); \
			install_filename=$(app_libdir)/$$filename; \
			install_name_tool -id @executable_path/$$filename $$install_filename; \
			install_name_tool -change $$j @executable_path/../../../lib/$$filename $$i; \
			codesign --force -s - $$install_filename ; \
		done; \
	done; \
	for copied_lib in $(AMBERLIB_SRC)/*.$(SHLIB_EXT); do \
		libname=`basename $$copied_lib`; \
		copy=$(app_libdir)/$$libname; \
		for k in `otool -L $$copied_lib | awk '(NF > 1 && $$1 ~ /^$(subst /,\/,$(FLIBDIR))/) { print $$1 }'`; do \
			subfilename=`basename $$k`; \
			install_name_tool -change $$k @executable_path/../../../lib/$$subfilename $$copy; \
			codesign --force -s - $$copy ; \
		done; \
	done;
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
	tar xf $(DISTRIBUTION) -C $(SOURCE) --no-same-owner
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(AMBERHOME) && patch -f -p0) < $$p ; \
	done
endif
## BUILD_TYPE is empty for local builds, otherwise one of daily, candidate, production
#ifeq ($(BUILD_TYPE),daily)
#	-cd $(AMBERHOME) && env AMBERHOME='$(AMBERHOME)' ./update_amber --update
#else
#	cd $(AMBERHOME) && env AMBERHOME='$(SOURCE)' ./update_amber --update
#endif
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
	rm -rf $(SOURCE) $(AMBERTOOLS_PREBUILT)
ifeq ($(OS),Darwin)
	rm -rf $(AMBERTOOLS_FLIBS_PREBUILT)
endif
