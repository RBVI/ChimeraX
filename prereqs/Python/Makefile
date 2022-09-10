PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# PYTHON_VERSION is in config.make
PYTHON_PATCH_LEVEL = $(PYTHON_VERSION).$(PYTHON_PATCH_VERSION)

PIP_VERSION = 22.2.2
WHEEL_VERSION = 0.37.1
SETUPTOOLS_VERSION = 65.1.1

ifeq ($(OS),Windows)
DISTRIBUTION = python-$(PYTHON_PATCH_LEVEL)-win-amd64.tar.bz2
DISTDIR = $(tmpdir)/python-$(PYTHON_PATCH_LEVEL)-win-amd64
SOURCE = 
DYNLOAD_DIR = $(bindir)/DLLs
endif
ifeq ($(OS),Linux)
DISTRIBUTION = Python-$(PYTHON_PATCH_LEVEL).tar.xz
DISTDIR =
SOURCE = $(tmpdir)/Python-$(PYTHON_PATCH_LEVEL)
DYNLOAD_DIR = $(libdir)/python$(PYTHON_VERSION)/lib-dynload
endif
ifeq ($(OS),Darwin)
ifeq ($(OSARCH),DarwinIntel64)
	DISTRIBUTION = python-$(PYTHON_PATCH_LEVEL)-mac.tar.bz2
else ifeq ($(OSARCH),DarwinArm64)
	DISTRIBUTION = python-$(PYTHON_PATCH_LEVEL)-mac-arm.tar.bz2
endif
DISTDIR =
SOURCE = 
DYNLOAD_DIR = $(libdir)/python$(PYTHON_VERSION)/lib-dynload
endif

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

# Make Python use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC) -I$(includedir) -L$(libdir) -DUSE_DYLD_GLOBAL_NAMESPACE'

ifdef DEBUG
ENV_CONFIGURE += OPT='$(OPT)'
endif

PATCHES	= $(wildcard Python-$(PYTHON_PATCH_LEVEL)-*.patch) \
	  $(wildcard Python-$(PYTHON_PATCH_LEVEL)_$(OS).patch) \
	  $(wildcard Python-$(PYTHON_PATCH_LEVEL)_$(OS)_$(OPENGL_PLATFORM).patch)
ifdef DEBUG
PATCHES += $(wildcard Python-$(PYTHON_PATCH_LEVEL)_debug.patch)
endif

ifeq ($(OS),Linux)
# -Wl,-rpath ends up in shared libraries in addition to executables,
# which we do not want, so only add the rpath when linking the executable
ENV_CONFIGURE += LDFLAGS='$(LDFLAGS)' LINKCC="gcc '-Wl,-rpath,\$$\$$ORIGIN/../lib'"
ifdef HARDENING
ENV_CONFIGURE += CFLAGS='-D_FORTIFY=2 -fstack-protector-strong'
endif
EXTRA_CFLAGS = '-fdebug-prefix-map=$(build_prefix)=.'
CONFIG_OPTIONS += --enable-shared
export LD_LIBRARY_PATH=$(libdir)
# --enable-optimizations causes singularity build to fail (and causes long build times)
# Try reneabling with newer versions of Python after benchmarking it is worth it.
# CONFIG_OPTIONS += --enable-optimizations
endif

ifeq ($(OS),Darwin)
# -Wl,-rpath ends up in shared libraries in addition to executables,
# which we do not want, so we add the rpath later, and set the DYLD_LIBRARY_PATH
# to compensate while we are building.  We also edit the Makefile so that the
# _build_ Python executable, python.exe, gets the correct rpath but none of
# the shared objects should get an explicit rpath (and hence should just use
# the rpath from the ChimeraX/python executable).
export DYLD_LIBRARY_PATH=$(libdir)
ENV_CONFIGURE += PATH='$(bindir)':"$$PATH"
ifndef USE_MAC_FRAMEWORKS
ENV_CONFIGURE += LDFLAGS='$(LDFLAGS)'
CONFIG_OPTIONS += --enable-shared
else
CONFIG_OPTIONS += --enable-framework=$(frameworkdir) --prefix=$(build_prefix)
ENV_CONFIGURE += LDFLAGS='$(LDFLAGS) -F$(frameworkdir)'
REL_PYTHON_FRAMEWORK = ../Library/Frameworks/Python.Framework/Versions/$(PYTHON_VERSION)
endif
endif

ifdef DEBUG
CONFIG_OPTIONS += --with-pydebug
endif

ifneq (,$(shell [ ! -e $(TOP)/build_type ] || [ `cat $(TOP)/build_type` = alpha ] && echo found))
# stuff to do only if it is a private or daily build
ifneq (,$(wildcard /usr/include/valgrind/valgrind.h))
CONFIG_OPTIONS += --with-valgrind
endif
endif

all:

install: $(SOURCE) $(DISTRIBUTION) $(DISTDIR)
ifeq ($(OS),Windows)
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' --exclude=__pycache__ --delete\
	  '$(DISTDIR)'/DLLs \
	  '$(DISTDIR)'/Lib \
	  '$(DISTDIR)'/include \
	  '$(DISTDIR)'/libs \
	  '$(DISTDIR)'/python.exe \
	  '$(DISTDIR)'/pythonw.exe \
	  '$(DISTDIR)'/*.dll \
	  '$(bindir)'
	# Leave include and libs in bindir so pip/setuptools will work.
	# Also copy include and lib directories to common locations for
	# our source code.
	$(RSYNC) --delete '$(DISTDIR)'/include/* \
		$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)
	$(RSYNC) '$(DISTDIR)'/libs/* $(libdir)

else ifdef USE_MAC_FRAMEWORKS
	rm -rf $(frameworkdir)/Python.framework
	tar xf $(DISTRIBUTION) -C $(frameworkdir)

	# Link framework bin, lib, and include files to app directories
	cd $(bindir) ; ln -s $(REL_PYTHON_FRAMEWORK)/bin/python* $(REL_PYTHON_FRAMEWORK)/bin/pip* .
	ln -s $(REL_PYTHON_FRAMEWORK)/lib/python$(PYTHON_VERSION) $(libdir)
	ln -s $(REL_PYTHON_FRAMEWORK)/include/python$(PYTHON_VERSION) $(includedir)

else
	# Linux or Mac OS X without framework
	cd $(SOURCE) && $(MAKE) && $(MAKE) install
ifeq ($(OS),Linux)
	-for lib in $(PYTHON_LIBRARY_DIR)/lib-dynload/*.so; do \
		$(bindir)/chrpath -d $$lib; \
	done
	chmod 755  $(libdir)/libpython*.so; $(bindir)/chrpath -d $(libdir)/libpython*.so
	# convert to RPATH
	if objdump -p $(PYTHON_BIN) | grep RUNPATH > /dev/null; then \
	       $(bindir)/chrpath -C $(PYTHON_BIN); \
	fi
endif
	$(RSYNC) $(SOURCE)/Misc/valgrind-python.supp $(datadir)
	sed -i -e 's;obj:.*libcrypto.*;obj:*/libcrypto.so.*;' $(datadir)/valgrind-python.supp
endif
	# remove unused libraries
	rm -rf $(PYTHON_LIBRARY_DIR)/config-*/*.{a,dylib}

	# Since pip comes with Python 3.5, update it now so installs on
	# Windows will work.  Wheel is needed so bdist_wheel will work.
	$(PYTHON_EXE) -m ensurepip --upgrade
	$(PYTHON_EXE) -m pip install --upgrade pip==$(PIP_VERSION)
	$(PYTHON_EXE) -m pip install --upgrade setuptools==$(SETUPTOOLS_VERSION)
	$(PYTHON_EXE) -m pip install --upgrade wheel==$(WHEEL_VERSION)

	# double check that _ssl was successfully made
	test -f $(DYNLOAD_DIR)/_ssl.*$(PYMOD_EXT)
	# double check that pyexpat was successfully made
	test -f $(DYNLOAD_DIR)/pyexpat.*$(PYMOD_EXT)

app-install: $(DISTDIR)
	$(RSYNC) $(PYTHON_BIN) $(app_bindir)
ifeq ($(OS),Windows)
	# need include and libs so pip install will work
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' --exclude=__pycache__ --delete\
	  '$(DISTDIR)'/DLLs \
	  '$(DISTDIR)'/Lib \
	  '$(DISTDIR)'/include \
	  '$(DISTDIR)'/libs \
	  '$(DISTDIR)'/*.dll \
	  $(app_bindir)
else ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(frameworkdir)/Python.framework $(app_frameworkdir)
	ln -s -f $(REL_PYTHON_FRAMEWORK)/bin/pip $(app_bindir)
	ln -s -f -F $(REL_PYTHON_FRAMEWORK)/lib/python$(PYTHON_VERSION) $(app_libdir)
	ln -s -f -F $(REL_PYTHON_FRAMEWORK)/include/python$(PYTHON_VERSION) $(app_includedir)
else
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' --exclude=__pycache__ \
		$(libdir)/libpython*.$(SHLIB_EXT)* \
		$(PYTHON_LIBRARY_DIR) $(app_libdir)
	$(RSYNC) $(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI) $(app_includedir)
	$(RSYNC) $(datadir)/valgrind-python.supp $(app_datadir)
	sed -i -e 's;obj:.*libcrypto.*;obj:*/libcrypto.so.*;' $(app_datadir)/valgrind-python.supp
endif
ifneq ($(OS),Windows)
	# remove references to build tree from config
	# we do not build Python on Windows, so no action is needed
	$(APP_PYTHON_EXE) clean_config_vars.py $(build_prefix)
endif
	$(APP_PYTHON_EXE) -m ensurepip --upgrade
	$(APP_PYTHON_EXE) -m pip install --upgrade pip==$(PIP_VERSION)
	$(APP_PYTHON_EXE) -m pip install --upgrade \
		wheel==$(WHEEL_VERSION) \
		setuptools==$(SETUPTOOLS_VERSION)

ifeq ($(OS),Windows)

$(DISTDIR): $(DISTRIBUTION)
	tar xf $(DISTRIBUTION) -C $(tmpdir)

else

$(SOURCE): $(DISTRIBUTION)
	tar -xf $(DISTRIBUTION) -C $(tmpdir)

	# avoid bootstrap (files generated by python, but python isn't built)
	cd $(SOURCE) && touch Python/Python-ast.c Include/Python-ast.h

ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
	cd $(SOURCE) && $(ENV_CONFIGURE) \
	    ./configure --prefix=$(build_prefix) $(CONFIG_OPTIONS)

ifdef USE_MAC_FRAMEWORKS
	# Remove this when we no longer support El Capitan (10.11)
	cd $(SOURCE) && \
		mv pyconfig.h pyconfig.h.old && \
		sed -e /HAVE_GETENTROPY/d \
			-e /HAVE_CLOCK_GETRES/d \
			-e /HAVE_CLOCK_GETTIME/d \
			-e /HAVE_CLOCK_SETTIME/d \
			< pyconfig.h.old > pyconfig.h
endif
	# Don't use stuff from /usr/local (especially Tcl, Tk, and SSL)
	cd $(SOURCE) && \
	    mv setup.py setup.py.orig ; \
	    sed -e 's;/usr/local;$(build_prefix);' -e 's;/ssl/;/;' < setup.py.orig > setup.py
ifdef USE_MAC_FRAMEWORKS
	# say where to find Tcl and Tk (and any other frameworks)
	(echo ':/framework_dirs =/s:$$: "$(frameworkdir)",:'; echo ':wq') \
	    | env TERM=dumb ex -s $(SOURCE)/setup.py
	(echo ':/^BASECFLAGS/s:$$: -F$(frameworkdir):'; echo ':wq') \
	    | env TERM=dumb ex -s $(SOURCE)/Makefile
	(echo ':/$$@ Programs\/python.o/s::& -Wl,-rpath,$(libdir):'; echo ':wq') \
	    | env TERM=dumb ex -s $(SOURCE)/Makefile
endif

endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/python/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/python

make_win_distribution:
	# only needed for Windows
	version="$$(regtool -w get /machine/SOFTWARE/PYTHON/PythonCore/$(PYTHON_VERSION)/Version)" && \
	exe_path="$$(regtool -w get /machine/SOFTWARE/PYTHON/PythonCore/$(PYTHON_VERSION)/InstallPath/ExecutablePath)" && \
	tar -jcf python-$$version-win-amd64.tar.bz2 --exclude __pycache__ --xform "s/^./python-$$version-win-amd64/" -C "$$exe_path"/.. .

make_mac_distribution:
	make -f Makefile.macos

clean:
	rm -rf $(SOURCE)
