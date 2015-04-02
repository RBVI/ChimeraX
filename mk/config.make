# need absolute directory for build_prefix
TOP := $(shell (cd "$(TOP)"; pwd))
build_prefix = $(TOP)/build
bindir = $(build_prefix)/bin
includedir = $(build_prefix)/include
libdir = $(build_prefix)/lib
shlibdir = $(libdir)
datadir = $(build_prefix)/share
tmpdir = $(build_prefix)/tmp
webdir = $(build_prefix)/webapp
wheelhouse = $(build_prefix)/wheelhouse

# by default, don't do anything
all:

# version numbers that leak out of prerequisites

PYTHON_VERSION = 3.4
ifdef DEBUG
PYTHON_ABI = dm
else
PYTHON_ABI = m
endif
# Windows uses python22.dll instead of libpython2.2.so
PYVER_NODOT = $(subst .,,$(PYTHON_VERSION))

include $(TOP)/mk/os.make

APP_NAME = Chimera2

ifeq ($(OS),Darwin)
frameworkdir = $(build_prefix)/Library/Frameworks
app_prefix = $(TOP)/$(APP_NAME).app/Contents
app_frameworkdir =  $(app_prefix)/Library/Frameworks
else
app_prefix = $(TOP)/$(APP_NAME).app
endif
app_bindir = $(app_prefix)/bin
app_includedir = $(app_prefix)/include
app_libdir = $(app_prefix)/lib
app_shlibdir = $(app_libdir)
app_datadir = $(app_prefix)/share

ifndef WIN32
RSYNC = rsync -rltWv --executability
else
RSYNC = $(bindir)/rsync.convert -rlptWv
endif

ifdef WIN32
PYTHON_INCLUDE_DIRS = -I'$(shell cygpath -m '$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)')'
PYTHON_LIBRARY_DIR = $(bindir)/Lib
APP_PYTHON_LIBRARY_DIR = $(app_bindir)/Lib
PYTHON_EXE = $(bindir)/python.exe
APP_PYTHON_EXE = $(app_bindir)/python.exe
else ifdef USE_MAC_FRAMEWORKS
PYTHON_INCLUDE_DIRS = $(shell $(bindir)/python$(PYTHON_VERSION)$(PYTHON_ABI)-config --includes)
PYTHON_FRAMEWORK = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)
APP_PYTHON_FRAMEWORK = $(app_frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)
PYTHON_LIBRARY_DIR = $(libdir)/python$(PYTHON_VERSION)
APP_PYTHON_LIBRARY_DIR = $(app_libdir)/python$(PYTHON_VERSION)
PYTHON_EXE = $(bindir)/python$(PYTHON_VERSION)
APP_PYTHON_EXE = $(app_bindir)/python$(PYTHON_VERSION)
else
PYTHON_INCLUDE_DIRS = -I$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)
PYTHON_LIBRARY_DIR = $(libdir)/python$(PYTHON_VERSION)
APP_PYTHON_LIBRARY_DIR = $(app_libdir)/python$(PYTHON_VERSION)
PYTHON_EXE = $(bindir)/python$(PYTHON_VERSION)
APP_PYTHON_EXE = $(app_bindir)/python$(PYTHON_VERSION)
endif
PYSITEDIR = $(PYTHON_LIBRARY_DIR)/site-packages
APP_PYSITEDIR = $(APP_PYTHON_LIBRARY_DIR)/site-packages

PYLINT = $(PYTHON_EXE) -I -m flake8
