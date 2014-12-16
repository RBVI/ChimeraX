FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# PYTHON_VERSION is in config.make
PYTHON_PATCH_LEVEL = $(PYTHON_VERSION).2
DISTRIBUTION = Python-$(PYTHON_PATCH_LEVEL).tar.xz
SOURCE = $(tmpdir)/Python-$(PYTHON_PATCH_LEVEL)

ifdef WIN32
DYNLOAD_DIR = $(bindir)/DLLs
else
DYNLOAD_DIR = $(libdir)/python$(PYTHON_VERSION)/lib-dynload
endif

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# Make Python use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC) -I$(includedir) -L$(libdir) -DUSE_DYLD_GLOBAL_NAMESPACE' CXX='$(QUOTE_CXX) -I$(includedir) -L$(libdir)'

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
# On linux have to disable IPV6 socket support or _socket doesn't compile.
CONFIG_OPTIONS += --disable-ipv6 --enable-shared
endif

ifndef USE_MAC_FRAMEWORKS
LDFLAGS += -Wl,-rpath,"$(libdir)"
ENV_CONFIGURE += LDFLAGS='$(LDFLAGS)'
else
LDFLAGS += -Wl,-rpath,"$(libdir)"
ENV_CONFIGURE += LDFLAGS='$(LDFLAGS) -F$(frameworkdir)'
CONFIG_OPTIONS += --enable-framework=$(frameworkdir) --exec_prefix=$(build_prefix)
pyappdir = $(PYTHON_FRAMEWORK)/Resources/Python.app/Contents/MacOS
endif

ifeq ($(OS),Windows)
ifeq ($(OSARCH),Windows64)
REGTOOL = regtool -w
CONFIG = Release
PLATFORM = x64
else
REGTOOL = regtool
CONFIG = Release
PLATFORM = Win32
endif
CONFIG_PLATFORM = '$(CONFIG)|$(PLATFORM)'

_PYDIR_MACHINE = /machine/SOFTWARE/Python/PythonCore/$(PYTHON_VERSION)/InstallPath/
_PYDIR_USER = /user/SOFTWARE/Python/PythonCore/$(PYTHON_VERSION)/InstallPath/
ifeq ($(shell $(REGTOOL) -q check $(_PYDIR_MACHINE) && echo found),found)
	PYDIR   = $(shell cygpath -u '$(shell $(REGTOOL) get $(_PYDIR_MACHINE))')
endif
ifeq ($(shell $(REGTOOL) -q check $(_PYDIR_USER) && echo found),found)
	PYDIR   = $(shell cygpath -u '$(shell $(REGTOOL) get $(_PYDIR_USER))')
endif
endif # Windows

ifdef DEBUG
CONFIG_OPTIONS += --with-pydebug
endif

ifneq (,$(shell [ ! -e $(TOP)/build_type ] || [ `cat $(TOP)/build_type` = alpha ] && echo found))
# stuff to do only if it is a private or daily build
ifneq (,$(wildcard /usr/include/valgrind/valgrind.h))
CONFIG_OPTIONS += --with-valgrind
endif
endif

# 64-bit Linux dies in exception handling if we don't link with the C++ compiler
CONFIG_OPTIONS += --with-cxx-main

all:

install: $(SOURCE)
	$(RSYNC) $(SOURCE)/Tools/freeze $(datadir)
ifeq ($(OS),Windows)
	#cd $(SOURCE)/PCbuild ; cmd /c build.bat -p $(PLATFORM) 
	#cd $(SOURCE)/PCbuild ; env HOST_PYTHON=`cygpath -m '$(SOURCE)/PCbuild/amd64/python3.exe'` devenv.com pcbuild.sln /build $(CONFIG_PLATFORM) /project _ssl
	test -n '$(PYDIR)'
	$(RSYNC) --include=*.exe --exclude='site-packages/*' \
	  $(PYDIR)/DLLs \
	  $(PYDIR)/Lib \
	  $(PYDIR)/include \
	  $(PYDIR)/libs \
	  $(PYDIR)/python.exe \
	  $(PYDIR)/pythonw.exe \
	  $(bindir)
else ifdef USE_MAC_FRAMEWORKS
	rm -rf $(frameworkdir)/Python.framework
	cd $(SOURCE) && $(MAKE) && $(MAKE) -j1 install
	rm -rf '$(build_prefix)/Applications/Python $(PYTHON_VERSION)'
	cp $(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)/pyconfig.h $(frameworkdir)/Python.framework/Headers/
	$(PYTHON_EXE) fixsymlinks.py $(PYTHON_FRAMEWORK)/lib/* \
		$(libdir)/python$(PYTHON_VERSION)/config-$(PYTHON_VERSION)$(PYTHON_ABI)/*
	chmod 644 $(PYTHON_FRAMEWORK)/Python
	install_name_tool -id \
		@executable_path/$(shell python -c 'import os; print(os.path.relpath("$(PYTHON_FRAMEWORK)/Python", "$(bindir)"))') \
		$(PYTHON_FRAMEWORK)/Python
	chmod 444 $(PYTHON_FRAMEWORK)/Python
# In Python 3.4 framework install only installs a python executable that uses Python.app in the framework.
	install_name_tool -rpath $(libdir) \
		'@executable_path/../lib' \
		$(bindir)/python$(PYTHON_VERSION)
	install_name_tool -change $(PYTHON_FRAMEWORK)/Python \
		@executable_path/$(shell python -c 'import os; print(os.path.relpath("$(PYTHON_FRAMEWORK)/Python", "$(bindir)"))') \
		$(bindir)/python$(PYTHON_VERSION)
	install_name_tool -rpath $(libdir)  \
		@executable_path/$(shell python -c 'import os; print(os.path.relpath("$(libdir)", "$(pyappdir)"))') \
		$(pyappdir)/Python
	install_name_tool -change $(PYTHON_FRAMEWORK)/Python \
		@executable_path/$(shell python -c 'import os; print(os.path.relpath("$(PYTHON_FRAMEWORK)/Python", "$(pyappdir)"))') \
		$(pyappdir)/Python
else
	# Linux or Mac OS X without framework
	cd $(SOURCE) && $(MAKE) && $(MAKE) install
ifeq ($(OS),Linux)
	chrpath -r '$$ORIGIN/../lib' $(bindir)/python$(PYTHON_VERSION) $(bindir)/python$(PYTHON_VERSION)$(PYTHON_ABI)
endif
ifeq ($(OS),Darwin)
	install_name_tool -rpath $(libdir) '@executable_path/../lib' $(bindir)/python$(PYTHON_VERSION)
endif
endif
	# double check that _ssl was successfully made
	test -f $(DYNLOAD_DIR)/_ssl.*$(PYMOD_EXT)
	# double check that pyexpat was successfully made
	test -f $(DYNLOAD_DIR)/pyexpat.*$(PYMOD_EXT)

app-install:
	$(RSYNC) $(PYTHON_EXE) $(app_bindir)
ifeq ($(OS),Windows)
	$(RSYNC) --exclude='site-packages/*'  --exclude='test/*'\
	  $(bindir)/DLLs \
	  $(bindir)/Lib \
	  $(app_bindir)
else ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(frameworkdir)/Python.framework $(app_frameworkdir)
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' \
		$(PYTHON_LIBRARY_DIR) $(app_libdir)
else
	$(RSYNC) --exclude='site-packages/*' --exclude='test/*' \
		$(libdir)/libpython*.$(SHLIB_EXT)* \
		$(PYTHON_LIBRARY_DIR) $(app_libdir)
endif

$(SOURCE):
ifneq ($(OS),Windows)
	$(bindir)/7za e -so $(DISTRIBUTION) | tar xf - -C $(tmpdir)

	# avoid bootstrap (files generated by python, but python isn't built)
	cd $(SOURCE) && touch Python/Python-ast.c Include/Python-ast.h

ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
	cd $(SOURCE) && $(ENV_CONFIGURE) \
	    ./configure --prefix=$(build_prefix) $(CONFIG_OPTIONS)
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
endif
endif

clean:
	rm -rf $(SOURCE)
