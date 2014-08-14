# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

# if CDPATH is set, then scripts that invoke cd will echo the new directory
# and potentially screw up constructed files.
unexport CDPATH

TOP	= ..
NO_SUBDIR_ALL = 1
NO_SUBDIR_DISTCLEAN = 1

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
SUBDIRS += leapmotion
endif

#ifdef UNIX
#SUBDIRS += unix
#endif

ifdef NEED_MESA
SUBDIRS += Mesa
else
Mesa.install:
endif

SUBDIRS	+= cython docutils expat ffmpeg fonts \
	   glew hdf5 imex jinja2 jpeg line_profiler numexpr numpy openssl \
	   p7zip pcre pillow pycollada \
	   pygments pyflakes PyOpenGL pytables pythomnic3k Python python-dateutil PyQt qt \
	   setuptools sip sphinx webcolors wrappy3 wx zlib

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	make -C $(TOP) build-dirs
	-mkdir $(tmpdir)
ifdef USE_MAC_FRAMEWORKS
	-mkdir $(includedir)/GL
	-$(RSYNC) /System/Library/Frameworks/OpenGL.framework/Headers/* \
	  $(includedir)/GL
endif

distclean: clean

ifeq ($(OS),Linux)
GCC_VER = $(shell $(CC) -dumpversion)
PREBUILT = prebuilt-$(OSARCH)-$(GCC_VER).7z
else ifeq ($(OS),Darwin)
PREBUILT = prebuilt-$(OSARCH)-$(MACOSX_DEPLOYMENT_TARGET).7z
else ifeq ($(OS),Windows)
PREBUILT = prebuilt-$(OSARCH).7z
else
PREBUILT = prebuilt-$(shell uname -s)-$(shell uname -m).7z
endif

prebuilt:
	#(cd $(build_prefix) && tar -cj --exclude ./tmp .) > $(PREBUILT)
	prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && 7za a -mx9 -x!./tmp $$prebuilt .

install.prebuilt:
	if [ -e $(PREBUILT) ]; then \
		$(MAKE) build-dirs ; \
		prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && 7za x $$prebuilt ; \
	else \
		$(MAKE) install ; \
	fi

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

blockdiag.install: Python.install funcparserlib.install webcolors.install
cython.install: Python.install
distribute.install: Python.install setuptools.install
docutils.install: Python.install pillow.install pygments.install
funcparserlib.install: Python.install
hdf5.install: zlib.install
jinja2.install: Python.install setuptools.install
leapmotion.install: Python.install
line_profiler.install: Python.install cython.install
numexpr.install: numpy.install
numpy.install: Python.install
pcre.install: zlib.install
pillow.install: Python.install setuptools.install zlib.install jpeg.install
# pillow.install: jpeg.install tiff.install freetype.install littlecms.install webp.install
pycollada.install: Python.install python-dateutil.install
pyflakes.install: Python.install
pygments.install: Python.install
PyOpenGL.install: Python.install numpy.install Mesa.install setuptools.install
pyside.install: cmake.install openssl.install xml2.install xslt.install pcre.install qt.install Python.install
pytables.install: hdf5.install numexpr.install cython.install Python.install zlib.install
pythomnic3k.install: Python.install
Python.install: p7zip.install zlib.install openssl.install expat.install
python-dateutil.install: Python.install
PyQt.install: Python.install qt.install sip.install
qt.install: p7zip.install openssl.install zlib.install
wx.install: Python.install
# qt.install: png.install jpeg.install
setuptools.install: Python.install
sip.install: Python.install
sphinx.install: docutils.install jinja2.install pygments.install setuptools.install
sphinx-contrib.install: sphinx.install blockdiag.install setuptools.install
webcolors.install: Python.install
wrappy3.install: pcre.install Python.install
xml2.install: Python.install zlib.install
xslt.install: xml2.install
