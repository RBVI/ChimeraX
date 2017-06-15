# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

# if CDPATH is set, then scripts that invoke cd will echo the new directory
# and potentially screw up constructed files.
unexport CDPATH

TOP	= ..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

all:
	@echo "'make install' to build everything" \
	&& echo "or 'make SUBIDR.install' to install that"

#
# Set your various NEEDs above.
#

NEED_FFMPEG=1
NEED_ZLIB=1

ifdef WIN32
SUBDIRS += win32
endif

ifeq ($(OS),Darwin)
NEED_EXPAT=1
NEED_HDF5=1
NEED_OPENSSL=1
NEED_P7ZIP=1
#SUBDIRS += leapmotion
SUBDIRS += oculus
endif

ifeq ($(OS),Linux)
NEED_EXPAT=1
NEED_HDF5=1
NEED_OPENSSL=1
NEED_P7ZIP=1
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

ifdef NEED_OFFSCREEN
SUBDIRS += llvm mesa
ifeq ($(OS),Darwin)
NEED_PKGCONFIG=1
endif
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
SUBDIRS	+= fonts \
	   imex numexpr numpy openmm \
	   pcre pdbfixer pillow pips pycollada \
	   PyOpenGL PyQt pytables Python \
	   scipy setuptools wrappy3

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	make -C $(TOP) build-dirs
	-mkdir $(tmpdir)

build-app-dirs:
	make -C $(TOP) build-app-dirs

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
# do Python first
#$(subst setuptools.app-install,,$(subst Python.app-install,,$(SUBDIR_APP_INSTALL))): setuptools.app-install
setuptools.app-install: Python.app-install

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
hdf5.app-install: zlib.app-install
hdf5.install: zlib.install
leapmotion.app-install: setuptools.app-install
leapmotion.install: setuptools.install
llvm.install: py7zip.install
mesa.install: py7zip.install llvm.install pkgconfig.install
numexpr.app-install: setuptools.app-install numpy.app-install pips.app-install
numexpr.install: setuptools.install numpy.install pips.install
openmm.app-install: Python.app-install numpy.app-install pips.app-install
openmm.install: Python.install numpy.install pips.install
pcre.app-install: zlib.app-install
pcre.install: zlib.install
pdbfixer.app-install: Python.app-install openmm.app-install
pdbfixer.install: Python.install openmm.install
pillow.app-install: setuptools.app-install zlib.app-install
pillow.install: setuptools.install zlib.install
pips.app-install: setuptools.app-install
pips.install: setuptools.install
pycollada.app-install: setuptools.app-install pips.app-install
pycollada.install: setuptools.install pips.install
PyOpenGL.app-install: setuptools.app-install pips.app-install
PyOpenGL.install: setuptools.install pips.install
PyQt.app-install: setuptools.app-install openssl.app-install zlib.app-install
PyQt.install: setuptools.install openssl.install zlib.install
pytables.app-install: hdf5.app-install numexpr.app-install pips.app-install setuptools.app-install zlib.app-install
pytables.install: hdf5.install numexpr.install pips.install setuptools.install zlib.install
pythomnic3k.app-install: setuptools.app-install
pythomnic3k.install: setuptools.install
Python.app-install: zlib.app-install openssl.app-install expat.app-install
Python.install: p7zip.install zlib.install openssl.install expat.install
scipy.app-install: setuptools.app-install numpy.app-install
scipy.install: setuptools.install numpy.install
setuptools.app-install: Python.app-install
setuptools.install: Python.install
wrappy3.app-install: pcre.app-install setuptools.app-install
wrappy3.install: pcre.install setuptools.install
wx.app-install: setuptools.app-install
wx.install: setuptools.install
xml2.app-install: setuptools.app-install zlib.app-install
xml2.install: setuptools.install zlib.install
xslt.app-install: xml2.app-install
xslt.install: xml2.install
