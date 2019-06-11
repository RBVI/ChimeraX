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
NEED_HDF5=1
NEED_OPENSSL=1
NEED_PKGCONFIG=1
NEED_P7ZIP=1
NEED_ZLIB=1
#SUBDIRS += leapmotion
SUBDIRS += oculus
endif

ifeq ($(OS),Linux)
NEED_CHRPATH=1
NEED_P7ZIP=1
NEED_MESA=1
NEED_HDF5=1
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
NEED_PKGCONFIG=1
NEED_LIBTOOL=1
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
SUBDIRS	+= imex lineprofiler numpy openmm \
	   pips \
	   PyOpenGL PyQt Python \
	   scipy tables

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	$(MAKE) -C $(TOP) build-dirs
	-mkdir $(tmpdir)

build-app-dirs:
	$(MAKE) -C $(TOP) build-app-dirs

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
ifndef UBUNTU_DIST
ifndef CENTOS_DIST
	prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar --exclude ./tmp -cjf $$prebuilt .
endif
endif

install-prebuilt:
	@echo 'Started prereq install at' `date` on `hostname`
	if [ -e $(PREBUILT) ]; then \
		$(MAKE) build-dirs ; \
		prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar xjf $$prebuilt ; \
	else \
		$(MAKE) install && $(MAKE) prebuilt; \
	fi
	@echo 'Finished prereq install at' `date`

SUBDIR_APP_INSTALL = $(SUBDIRS:=.app-install)
.PHONY: $(SUBDIR_APP_INSTALL)
app-install: $(SUBDIR_APP_INSTALL)
#	# recreate $(PYSITEDIR)/easy-install.pth with actually installed eggs
#	cd $(APP_PYSITEDIR); \
#		(echo "import sys; sys.__plen = len(sys.path)"; \
#		ls -d ./*egg; \
#		echo "import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)") > easy-install.pth

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

ffmpeg.install: p7zip.install pkgconfig.install libtool.install
hdf5.app-install: zlib.app-install chrpath.app-install
hdf5.install: zlib.install chrpath.install
leapmotion.app-install: Python.app-install
leapmotion.install: Python.install
lineprofiler.install: pips.install
lineprofiler.app-install: pips.app-install
llvm.install: p7zip.install
mesa.install: p7zip.install llvm.install pkgconfig.install
msgpack.app-install: Python.app-install pips.app-install
msgpack.install: Python.install pips.install
numpy.app-install: Python.app-install numpy.app-install pips.app-install
numpy.install: Python.install numpy.install pips.install
openmm.app-install: Python.app-install numpy.app-install pips.app-install
openmm.install: Python.install numpy.install pips.install
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
pythomnic3k.app-install: Python.app-install
pythomnic3k.install: Python.install
Python.app-install: zlib.app-install openssl.app-install expat.app-install chrpath.app-install
Python.install: p7zip.install zlib.install openssl.install expat.install chrpath.install
scipy.app-install: Python.app-install numpy.app-install
scipy.install: Python.install numpy.install
# tables could also need Cython, numexpr, and c-blosc
tables.install: hdf5.install pips.install numpy.install
tables.app-install: hdf5.app-install pips.app-install numpy.app-install
xml2.app-install: Python.app-install zlib.app-install
xml2.install: Python.install zlib.install
xslt.app-install: xml2.app-install
xslt.install: xml2.install
