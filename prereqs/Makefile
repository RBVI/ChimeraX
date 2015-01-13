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

# Python egg installation has a race condition updating easy-install.pth
.NOTPARALLEL:

# keep in alphabetical order, dependencies go below
SUBDIRS	+= appdirs beautifulsoup cython docutils expat flake8 ffmpeg fonts \
	   glew hdf5 imex jinja2 jpeg line_profiler numexpr numpy numpydoc \
	   openssl ordered_set p7zip pcre pillow pycollada \
	   pygments pyflakes PyOpenGL pytables pythomnic3k Python \
	   python-dateutil \
	   setuptools six sphinx webcolors wrappy3 wx zlib

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
	# recreate $(PYSITEDIR)/easy-install.pth with actually installed eggs
	cd $(APP_PYSITEDIR); \
		(echo "import sys; sys.__plen = len(sys.path)"; \
		ls -d ./*egg; \
		echo "import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)") > easy-install.pth
$(SUBDIR_APP_INSTALL):
	$(MAKE) -C $(subst .app-install,,$@) app-install
$(SUBDIR_APP_INSTALL): build-app-dirs
# do Python first
$(subst Python.app-install,,$(SUBDIR_APP_INSTALL)): Python.app-install


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

appdirs.install: Python.install setuptools.install
beautifulsoup.install: Python.install setuptools.install
blockdiag.install: Python.install funcparserlib.install webcolors.install
cython.install: Python.install
distribute.install: Python.install setuptools.install
docutils.install: Python.install pillow.install pygments.install
flake8.install: Python.install
funcparserlib.install: Python.install
hdf5.install: zlib.install
jinja2.install: Python.install setuptools.install
leapmotion.install: Python.install
line_profiler.install: Python.install cython.install
numexpr.install: numpy.install
numpy.install: Python.install
numpydoc.install: Python.install setuptools.install
ordered_set.install: Python.install setuptools.install
pcre.install: zlib.install
# pillow.install: jpeg.install tiff.install freetype.install littlecms.install webp.install
pillow.install: Python.install setuptools.install zlib.install jpeg.install
pycollada.install: Python.install python-dateutil.install numpy.install
pyflakes.install: Python.install
pygments.install: Python.install
PyOpenGL.install: Python.install numpy.install Mesa.install setuptools.install
PyQt.install: Python.install qt.install sip.install
pyside.install: cmake.install openssl.install xml2.install xslt.install pcre.install qt.install Python.install
pytables.install: hdf5.install numexpr.install cython.install Python.install zlib.install
pythomnic3k.install: Python.install
python-dateutil.install: Python.install six.install
Python.install: p7zip.install zlib.install openssl.install expat.install
qt.install: p7zip.install openssl.install zlib.install
# qt.install: png.install jpeg.install
setuptools.install: Python.install
sip.install: Python.install
six.install: Python.install
sphinx-contrib.install: sphinx.install blockdiag.install setuptools.install
sphinx.install: docutils.install jinja2.install numpydoc.install pygments.install setuptools.install
webcolors.install: Python.install
wrappy3.install: pcre.install Python.install
wx.install: Python.install
xml2.install: Python.install zlib.install
xslt.install: xml2.install
