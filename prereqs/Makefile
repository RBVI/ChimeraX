# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

# if CDPATH is set, then scripts that invoke cd will echo the new directory
# and potentially screw up constructed files.
unexport CDPATH

TOP	= ..
NO_SUBDIR_ALL = 1
NO_SUBDIR_TEST = 1

include $(TOP)/mk/config.make

all:
	@echo "'make install' to build everything" \
	&& echo "or 'make SUBIDR.install' to install that"

#
# Set your various NEEDs above.
#

ifndef UBUNTU_DIST
NEED_FFMPEG=1
endif

ifdef WIN32
SUBDIRS += win32
endif

ifeq ($(OS),Darwin)
NEED_EXPAT=1
# NEED_HDF5=1
NEED_OPENSSL=1
NEED_PKGCONFIG=1
NEED_P7ZIP=1
NEED_ZLIB=1
endif

ifeq ($(OS),Linux)
NEED_CHRPATH=1
NEED_P7ZIP=1
ifndef CENTOS_DIST
NEED_MESA=1
endif
ifeq ($(CENTOS_DIST),7)
NEED_MESA=1
endif
# NEED_HDF5=1
ifndef UBUNTU_DIST
ifndef CENTOS_DIST
NEED_EXPAT=1
NEED_OPENSSL=1
NEED_ZLIB=1
endif
endif
endif

ifdef NEED_CHRPATH
SUBDIRS += chrpath
else
chrpath.install:
chrpath.app-install:
endif

ifdef NEED_FFMPEG
SUBDIRS += ffmpeg
ifeq ($(OS),Darwin)
NEED_LIBTOOL=1
NEED_PKGCONFIG=1
endif
else
ffmpeg.install:
ffmpeg.app-install:
endif

ifdef NEED_MESA
SUBDIRS += mesa
ifeq ($(OS),Darwin)
NEED_PKGCONFIG=1
endif
else
mesa.install:
mesa.app-install:
endif

ifdef NEED_LLVM
SUBDIRS += llvm
ifeq ($(OS),Darwin)
NEED_PKGCONFIG=1
endif
else
llvm.install:
llvm.app-install:
endif

ifdef NEED_LIBTOOL
SUBDIRS += libtool
else
libtool.install:
libtool.app-install:
endif

ifdef NEED_PKGCONFIG
SUBDIRS += pkgconfig
else
pkgconfig.install:
pkgconfig.app-install:
endif

ifdef NEED_EXPAT
SUBDIRS += expat
else
expat.install:
expat.app-install:
endif

ifdef NEED_HDF5
SUBDIRS += hdf5
else
hdf5.install:
hdf5.app-install:
endif

ifdef NEED_OPENSSL
SUBDIRS += openssl
else
openssl.install:
openssl.app-install:
endif

ifdef NEED_P7ZIP
SUBDIRS += p7zip
else
p7zip.install:
p7zip.app-install:
endif

ifdef NEED_ZLIB
SUBDIRS += zlib
else
zlib.install:
zlib.app-install:
endif

# Python egg installation has a race condition updating easy-install.pth
.NOTPARALLEL:

# keep in alphabetical order, dependencies go below
SUBDIRS	+= cxservices gdcm imex lineprofiler openmm \
	   pips PyOpenGL PyQt Python

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	$(MAKE) -C $(TOP) build-dirs
	-mkdir $(tmpdir)

build-app-dirs:
	$(MAKE) -C $(TOP) build-app-dirs

ifdef NO_PREBUILT
prebuilt:

install-prebuilt:

else
ifeq ($(OS),Linux)
GCC_VER = $(shell $(CC) -dumpversion)
PREBUILT = prebuilt-$(OSARCH)-$(GCC_VER).tar.bz2
else ifeq ($(OS),Darwin)
PREBUILT = prebuilt-$(OSARCH)-$(MACOSX_DEPLOYMENT_TARGET).tar.bz2
else ifeq ($(OS),Windows)
PREBUILT = prebuilt-$(OSARCH).tar.bz2
else
PREBUILT = prebuilt-$(shell uname -s)-$(shell uname -m).tar.bz2
endif

prebuilt:
	prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar --exclude ./tmp -cjf $$prebuilt .

install-prebuilt:
	@echo 'Started prereq install at' `date` on `hostname`
	if [ -e $(PREBUILT) ]; then \
		$(MAKE) build-dirs ; \
		prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar xjf $$prebuilt ; \
	else \
		$(MAKE) install && $(MAKE) prebuilt; \
	fi
	@echo 'Finished prereq install at' `date`
endif

SUBDIR_APP_INSTALL = $(SUBDIRS:=.app-install)
.PHONY: $(SUBDIR_APP_INSTALL)
app-install: $(SUBDIR_APP_INSTALL)

$(SUBDIR_APP_INSTALL):
	$(MAKE) -C $(subst .app-install,,$@) app-install
$(SUBDIR_APP_INSTALL): build-app-dirs

# Whole platform dependencies
ifdef WIN32
$(subst win32.install,,$(SUBDIR_INSTALL)): win32.install
endif
#ifdef UNIX
#$(subst unix.install,,$(SUBDIR_INSTALL)): unix.install
#endif

# Please keep dependencies in alphabetical order.
# All packages that have dependencies should depend on the .install
# target of the other packages.

cxservices.install: pips.install
cxservices.app-install: pips.app-install
ffmpeg.install: p7zip.install pkgconfig.install libtool.install
gdcm.app-install: Python.app-install
gdcm.install: Python.install
hdf5.app-install: zlib.app-install chrpath.app-install
hdf5.install: zlib.install chrpath.install
libtool.install: PyQt.install  # when building from source Qt needs the system libtool
lineprofiler.install: pips.install
lineprofiler.app-install: pips.app-install
llvm.install: p7zip.install
mesa.install: p7zip.install llvm.install pkgconfig.install
openmm.app-install: Python.app-install pips.app-install
openmm.install: Python.install pips.install
openssl.app-install: chrpath.app-install
openssl.install: chrpath.install pkgconfig.install
pillow.app-install: Python.app-install zlib.app-install
pillow.install: Python.install zlib.install
pips.app-install: Python.app-install
pips.install: Python.install
PyOpenGL.app-install: Python.app-install pips.app-install
PyOpenGL.install: Python.install pips.install
PyQt.app-install: Python.app-install openssl.app-install zlib.app-install
PyQt.install: Python.install openssl.install zlib.install
Python.app-install: zlib.app-install openssl.app-install expat.app-install chrpath.app-install
Python.install: p7zip.install zlib.install openssl.install expat.install chrpath.install
