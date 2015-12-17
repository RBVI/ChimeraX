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

## do we need Mesa? TODO: want tesselator from libGLU
#ifdef WIN32
## use prebuilt Python
#NEED_MESA=1
#else
#ifneq ($(OS),Darwin)
#NEED_MESA=1
#endif
#endif

#
# Set your various NEEDs above.
#
ifdef WIN32
SUBDIRS += win32
endif

ifeq ($(OS),Darwin)
NEED_OPENSSL=1
#SUBDIRS += leapmotion
SUBDIRS += oculus
endif

ifeq ($(OS),Linux)
NEED_OPENSSL=1
SUBDIRS += llvm mesa pkgconfig xdg-utils
endif

#ifdef UNIX
#SUBDIRS += unix
#endif

ifdef NEED_OPENSSL
SUBDIRS += openssl
else
openssl.install:
openssl.app-install:
endif

ifdef HYDRA
SUBDIRS += qt sip PyQt
endif

# Python egg installation has a race condition updating easy-install.pth
.NOTPARALLEL:

# keep in alphabetical order, dependencies go below
SUBDIRS	+= expat ffmpeg fonts \
	   glew hdf5 imex jpeg numexpr numpy \
	   p7zip pcre pips pycollada \
	   PyOpenGL pytables pythomnic3k Python \
	   setuptools wrappy3 wx zlib

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	make -C $(TOP) build-dirs
	-mkdir $(tmpdir)
ifeq ($(OS),Darwin)
	-mkdir $(includedir)/GL
	-$(RSYNC) /System/Library/Frameworks/OpenGL.framework/Headers/* \
	  $(includedir)/GL
endif

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

hdf5.install: zlib.install
hdf5.app-install: zlib.app-install
leapmotion.install: setuptools.install
leapmotion.app-install: setuptools.app-install
llvm.install:
mesa.install: llvm.install pkgconfig.install
numexpr.install: numpy.install
numexpr.app-install: numpy.app-install
numpy.install: setuptools.install
numpy.app-install: setuptools.app-install
pcre.install: zlib.install
pcre.app-install: zlib.app-install
# For Pillow:
# pips.install: jpeg.install tiff.install freetype.install littecms.install webp.install webpmux.install
# pips.app-install: jpeg.app-install tiff.app-install freetype.app-install littecms.app-install webp.app-install webpmux.app-install
pips.install: setuptools.install
pips.app-install: setuptools.app-install
pkgconfig.install:
pycollada.install: setuptools.install numpy.install pips.install
pycollada.app-install: setuptools.app-install numpy.app-install pips.app-install
PyOpenGL.install: setuptools.install numpy.install pips.install
PyOpenGL.app-install: setuptools.app-install numpy.app-install pips.app-install
PyQt.install: setuptools.install qt.install sip.install
PyQt.app-install: setuptools.app-install qt.app-install sip.app-install
pytables.install: hdf5.install numexpr.install pips.install setuptools.install zlib.install
pytables.app-install: hdf5.app-install numexpr.app-install pips.app-install setuptools.app-install zlib.app-install
pythomnic3k.install: setuptools.install
pythomnic3k.app-install: setuptools.app-install
Python.install: p7zip.install zlib.install openssl.install expat.install
Python.app-install: p7zip.app-install zlib.app-install openssl.app-install expat.app-install
qt.install: p7zip.install openssl.install zlib.install
qt.app-install: p7zip.app-install openssl.app-install zlib.app-install
# qt.install: png.install jpeg.install
# qt.app-install: png.app-install jpeg.app-install
setuptools.install: Python.install
setuptools.app-install: Python.app-install
sip.install: setuptools.install
sip.app-install: setuptools.app-install
wrappy3.install: pcre.install setuptools.install
wrappy3.app-install: pcre.app-install setuptools.app-install
wx.install: setuptools.install
wx.app-install: setuptools.app-install
xml2.install: setuptools.install zlib.install
xml2.app-install: setuptools.app-install zlib.app-install
xslt.install: xml2.install
xslt.app-install: xml2.app-install
