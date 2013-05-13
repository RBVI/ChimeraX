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
#ifdef UNIX
#SUBDIRS += unix
#endif

ifdef NEED_MESA
SUBDIRS += Mesa
else
Mesa.install:
endif

SUBDIRS	+= docutils distribute expat fonts glew jinja2 numpy openssl \
	   pcre PyOpenGL pyflakes Python PyQt qt \
	   sip sphinx wrappy3 zlib

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
PREBUILT = prebuilt-$(OSARCH)-$(GCC_VER).tar.bz2
else ifeq ($(OS),Darwin)
PREBUILT = prebuilt-$(OSARCH)-$(MACOSX_DEPLOYMENT_TARGET).tar.bz2
else ifeq ($(OS),Windows)
PREBUILT = prebuilt-$(OSARCH).tar.bz2
else
PREBUILT = prebuilt-$(shell uname -s)-$(shell uname -m).tar.bz2
endif

prebuilt:
	(cd $(build_prefix) && tar -cj --exclude ./tmp .) > $(PREBUILT)

install.prebuilt:
	if [ -e $(PREBUILT) ]; then \
		(cd $(build_prefix) && tar -xj) < $(PREBUILT) ; \
		$(MAKE) build-dirs ; \
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
distribute.install: Python.install
docutils.install: Python.install
funcparserlib.install: Python.install
jinja2.install: Python.install distribute.install
numpy.install: Python.install
pcre.install: zlib.install
pyflakes.install: Python.install
PyOpenGL.install: Python.install numpy.install Mesa.install distribute.install
pyside.install: cmake.install openssl.install xml2.install xslt.install pcre.install qt.install Python.install
Python.install: zlib.install openssl.install expat.install
PyQt.install: Python.install qt.install
qt.install: openssl.install zlib.install
# qt.install: png.install jpeg.install
sip.install: Python.install
sphinx.install: docutils.install jinja2.install distribute.install
sphinx-contrib.install: sphinx.install blockdiag.install distribute.install
webcolors.install: Python.install
wrappy3.install: pcre.install Python.install
xml2.install: Python.install zlib.install
xslt.install: xml2.install
