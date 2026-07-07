PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# PYTHON_VERSION and PYTHON_PATCH_VERSION are in config.make
PYTHON_PATCH_LEVEL = $(PYTHON_VERSION).$(PYTHON_PATCH_VERSION)

PIP_VERSION = 26.1.2
WHEEL_VERSION = 0.47.0
SETUPTOOLS_VERSION = 82.0.1
BUILD_VERSION = 1.5.0

# macOS and Windows use relocatable distributions from python-build-standalone
# (https://github.com/astral-sh/python-build-standalone), so no per-platform
# building is needed there.  Linux is built from CPython source (python.org) so
# that it dynamically links the distribution's system OpenSSL; the expensive
# build result is packaged into a tarball that CI caches per-distro.

ifeq ($(OS),Linux)
# --- Linux: build from source, link the system OpenSSL ---------------------
SOURCE_DIST = Python-$(PYTHON_PATCH_LEVEL).tar.xz
SOURCE = $(tmpdir)/Python-$(PYTHON_PATCH_LEVEL)
# Relocatable tarball of the freshly built interpreter (members relative to
# build_prefix).  CI caches this so the source build only runs on a cache miss.
# The per-distro identity lives in the CI cache key, not in this file name.
LINUX_PREBUILT = python-$(PYTHON_PATCH_LEVEL)-$(OSARCH).tar.bz2

CONFIG_OPTIONS += --enable-shared
# --enable-optimizations (PGO+LTO) is left off: it greatly lengthens the build
# and has broken some container builds.  Now that the result is cached, it could
# be reconsidered after benchmarking.
QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
ENV_CONFIGURE = env CC='$(QUOTE_CC) -I$(includedir) -L$(libdir)' LDFLAGS='$(LDFLAGS)'
ifdef HARDENING
ENV_CONFIGURE += CFLAGS='-D_FORTIFY=2 -fstack-protector-strong'
endif
export LD_LIBRARY_PATH=$(libdir)
ifdef DEBUG
CONFIG_OPTIONS += --with-pydebug
endif
ifneq (,$(shell [ ! -e $(TOP)/build_type ] || [ `cat $(TOP)/build_type` = alpha ] && echo found))
ifneq (,$(wildcard /usr/include/valgrind/valgrind.h))
CONFIG_OPTIONS += --with-valgrind
endif
endif
SRC_PATCHES = $(wildcard Python-$(PYTHON_PATCH_LEVEL)-*.patch) \
	$(wildcard Python-$(PYTHON_PATCH_LEVEL)_$(OS).patch)

else
# --- macOS / Windows: python-build-standalone -------------------------------
# Map our OSARCH to a python-build-standalone target triple.
ifeq ($(OSARCH),DarwinArm64)
PBS_TRIPLE = aarch64-apple-darwin
endif
ifeq ($(OSARCH),DarwinIntel64)
PBS_TRIPLE = x86_64-apple-darwin
endif
ifeq ($(OSARCH),Windows64)
PBS_TRIPLE = x86_64-pc-windows-msvc
endif
ifeq ($(OSARCH),WindowsArm64)
PBS_TRIPLE = aarch64-pc-windows-msvc
endif

DISTRIBUTION = cpython-$(PYTHON_PATCH_LEVEL)+$(PYTHON_BUILD_STANDALONE_RELEASE)-$(PBS_TRIPLE)-install_only.tar.gz
# The archive unpacks to a top-level "python" directory.
PBS_DIR = $(tmpdir)/python
INSTALL_DEP = $(PBS_DIR)
endif

ifeq ($(OS),Windows)
APP_INSTALL_DEP = $(PBS_DIR)
endif

ifeq ($(OS),Darwin)
# Path from bindir/libdir/includedir (build/bin, build/lib, build/include) to
# the framework, used for the convenience symlinks.
REL_PYTHON_FRAMEWORK = ../Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)
endif

all:

install: $(INSTALL_DEP)
ifeq ($(OS),Windows)
	# Windows layout: python.exe plus DLLs/Lib/include/libs at the top level.
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' --exclude=__pycache__ --delete\
	  '$(PBS_DIR)'/DLLs \
	  '$(PBS_DIR)'/Lib \
	  '$(PBS_DIR)'/include \
	  '$(PBS_DIR)'/libs \
	  '$(PBS_DIR)'/python.exe \
	  '$(PBS_DIR)'/pythonw.exe \
	  '$(PBS_DIR)'/*.dll \
	  '$(bindir)'
	# Leave include and libs in bindir so pip/setuptools will work.
	# Also copy include and lib directories to common locations for
	# our source code.
	$(RSYNC) --delete '$(PBS_DIR)'/include/* \
		$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)
	$(RSYNC) '$(PBS_DIR)'/libs/* $(libdir)

else ifeq ($(OS),Linux)
	# Use the cached prebuilt interpreter if present, otherwise build it from
	# CPython source (which links the distribution's system OpenSSL) and package
	# the result so CI can cache it.
	if [ -f '$(LINUX_PREBUILT)' ]; then \
		echo 'Unpacking cached Python build: $(LINUX_PREBUILT)' ; \
		tar xjf '$(LINUX_PREBUILT)' -C $(build_prefix) --no-same-owner ; \
	else \
		$(MAKE) linux-source-build ; \
	fi

else
	# macOS: repackage the relocatable distribution into a Python.framework so
	# the rest of the macOS build, code-signing, and packaging are unchanged.
	# The distribution's lib/ subtree is kept intact, so libpython and the
	# extension modules retain their @rpath/libpython$(PYTHON_VERSION).dylib
	# install names; the ChimeraX executable adds a matching rpath that points
	# at this framework's Versions/$(PYTHON_VERSION)/lib directory.
	rm -rf $(frameworkdir)/Python.framework
	mkdir -p $(PYTHON_FRAMEWORK)
	$(RSYNC) '$(PBS_DIR)'/bin '$(PBS_DIR)'/lib '$(PBS_DIR)'/include $(PYTHON_FRAMEWORK)
	-[ -d '$(PBS_DIR)'/share ] && $(RSYNC) '$(PBS_DIR)'/share $(PYTHON_FRAMEWORK)
	# framework entry points (the dylib is reached through these symlinks)
	ln -sf lib/libpython$(PYTHON_VERSION).dylib $(PYTHON_FRAMEWORK)/Python
	ln -sf include/python$(PYTHON_VERSION) $(PYTHON_FRAMEWORK)/Headers
	mkdir -p $(PYTHON_FRAMEWORK)/Resources
	sed -e 's/@PYTHON_PATCH_LEVEL@/$(PYTHON_PATCH_LEVEL)/g' \
	    Info.plist.in > $(PYTHON_FRAMEWORK)/Resources/Info.plist
	cd $(frameworkdir)/Python.framework/Versions && ln -sfn $(PYTHON_VERSION) Current
	cd $(frameworkdir)/Python.framework && ln -sfn Versions/Current/Python Python
	cd $(frameworkdir)/Python.framework && ln -sfn Versions/Current/Headers Headers
	cd $(frameworkdir)/Python.framework && ln -sfn Versions/Current/Resources Resources
	# Link framework bin, lib, and include into the build directories.
	-cd $(bindir) ; ln -sf $(REL_PYTHON_FRAMEWORK)/bin/python* $(REL_PYTHON_FRAMEWORK)/bin/pip* .
	-ln -sfn $(REL_PYTHON_FRAMEWORK)/lib/python$(PYTHON_VERSION) $(libdir)/python$(PYTHON_VERSION)
	-ln -sfn $(REL_PYTHON_FRAMEWORK)/include/python$(PYTHON_VERSION) $(includedir)/python$(PYTHON_VERSION)
endif
	# remove unused static libraries
	-rm -rf $(PYTHON_LIBRARY_DIR)/config-*/*.{a,dylib}

	# Update pip, setuptools, wheel, and build to our pinned versions.  pip
	# uses its own vendored certificates for these downloads, so this works
	# before certifi is installed by prereqs/pips.
	$(PYTHON_EXE) -m ensurepip --upgrade
	$(PYTHON_EXE) -m pip install --upgrade pip==$(PIP_VERSION)
	$(PYTHON_EXE) -m pip install --upgrade setuptools==$(SETUPTOOLS_VERSION)
	$(PYTHON_EXE) -m pip install --upgrade wheel==$(WHEEL_VERSION)
	$(PYTHON_EXE) -m pip install --upgrade build==$(BUILD_VERSION)

	# double check that ssl and expat support are present
	$(PYTHON_EXE) -c 'import ssl, pyexpat'
ifeq ($(OS),Darwin)
	# Rewrite the build tree's recorded toolchain (python-build-standalone bakes
	# in absolute paths to its hermetic build tools) so packages that compile
	# against this Python can find the system compiler and archiver.  The Linux
	# source build records the real system toolchain, so it needs no rewriting.
	$(PYTHON_EXE) clean_config_vars.py $(build_prefix)
endif

# Build CPython from source and package the result for caching (Linux only).
linux-source-build: $(SOURCE)
	cd $(SOURCE) && $(MAKE) && $(MAKE) install
	# Convert absolute rpaths to $$ORIGIN-relative so the build is relocatable.
	patchelf --remove-rpath $(PYTHON_LIBRARY_DIR)/lib-dynload/*.so
	chmod 755 $(libdir)/libpython*.so; patchelf --remove-rpath $(libdir)/libpython*.so
	patchelf --force-rpath --set-rpath '$$ORIGIN/../lib' $(PYTHON_BIN)
	$(RSYNC) $(SOURCE)/Misc/valgrind-python.supp $(datadir)
	sed -i -e 's;obj:.*libcrypto.*;obj:*/libcrypto.so.*;' $(datadir)/valgrind-python.supp
	# Package the freshly built interpreter (without pip, which the install
	# target installs fresh afterwards) so CI can cache it.
	cd $(build_prefix) && tar cjf '$(abspath $(LINUX_PREBUILT))' \
		bin/python$(PYTHON_VERSION) bin/python3 \
		lib/libpython*.so* \
		lib/python$(PYTHON_VERSION) \
		include/python$(PYTHON_VERSION)$(PYTHON_ABI) \
		share/valgrind-python.supp

app-install: $(APP_INSTALL_DEP)
	$(RSYNC) $(PYTHON_BIN) $(app_bindir)
ifeq ($(OS),Windows)
	# need include and libs so pip install will work
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' --exclude=__pycache__ --delete\
	  '$(PBS_DIR)'/DLLs \
	  '$(PBS_DIR)'/Lib \
	  '$(PBS_DIR)'/include \
	  '$(PBS_DIR)'/libs \
	  '$(PBS_DIR)'/*.dll \
	  $(app_bindir)
else ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(frameworkdir)/Python.framework $(app_frameworkdir)
	ln -s -f $(REL_PYTHON_FRAMEWORK)/bin/pip $(app_bindir)
	ln -s -f -n $(REL_PYTHON_FRAMEWORK)/lib/python$(PYTHON_VERSION) $(app_libdir)
	ln -s -f -n $(REL_PYTHON_FRAMEWORK)/include/python$(PYTHON_VERSION) $(app_includedir)
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
		setuptools==$(SETUPTOOLS_VERSION) \
		build==$(BUILD_VERSION)

# --- macOS / Windows: fetch and unpack the python-build-standalone archive ---
$(PBS_DIR): $(DISTRIBUTION)
	rm -rf $(PBS_DIR)
	tar -xf $(DISTRIBUTION) -C $(tmpdir) --no-same-owner

$(DISTRIBUTION):
	$(FETCH_PYTHON) $(PYTHON_BUILD_STANDALONE_BASE)/$(PYTHON_BUILD_STANDALONE_RELEASE)/$(DISTRIBUTION)

# --- Linux: fetch and configure the CPython source ---------------------------
$(SOURCE): $(SOURCE_DIST)
	tar -xf $(SOURCE_DIST) -C $(tmpdir) --no-same-owner
	# avoid bootstrap (files generated by python, but python isn't built)
	cd $(SOURCE) && touch Python/Python-ast.c Include/Python-ast.h
ifneq (,$(SRC_PATCHES))
	for p in $(SRC_PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
	cd $(SOURCE) && $(ENV_CONFIGURE) \
	    ./configure --prefix=$(build_prefix) $(CONFIG_OPTIONS)
	# Don't use stuff from /usr/local (especially Tcl, Tk, and SSL)
	cd $(SOURCE) && if [ -f setup.py ]; then \
	    mv setup.py setup.py.orig && \
	    sed -e 's;/usr/local;$(build_prefix);' -e 's;/ssl/;/;' < setup.py.orig > setup.py ; \
	fi

$(SOURCE_DIST):
	$(FETCH_PYTHON) $(PYTHON_SOURCE_BASE)/$(PYTHON_PATCH_LEVEL)/$(SOURCE_DIST)

clean:
	-rm -rf $(PBS_DIR) $(DISTRIBUTION) $(SOURCE) $(SOURCE_DIST) $(LINUX_PREBUILT)
