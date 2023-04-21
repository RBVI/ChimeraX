# -*- mode: makefile -*- vim: set syntax=make:
# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# bash is the default shell on Cygwin and Mac OS X so use it on Linux as well
SHELL=/bin/bash

# need absolute directory for build_prefix
TOP := $(shell (cd "$(TOP)"; pwd))
build_prefix = $(TOP)/build
bindir = $(build_prefix)/bin
includedir = $(build_prefix)/include
libdir = $(build_prefix)/lib
shlibdir = $(libdir)
datadir = $(build_prefix)/share
tmpdir = $(build_prefix)/tmp

export PKG_CONFIG_PATH=$(libdir)/pkgconfig

# by default, don't do anything
all:

# version numbers that leak out of prerequisites

PYTHON_VERSION = 3.9
PYTHON_PATCH_VERSION = 11

ifndef DEBUG
# Starting with Python 3.8 the ABI "m" has been dropped.
PYTHON_ABI = 
else
ifneq (,$(wildcard $(includedir)/python$(PYTHON_VERSION)d))
PYTHON_ABI = d
else
# didn't find debug include files, Python not compiled with debugging,
# so revert to regular Python ABI
PYTHON_ABI = 
endif
endif
# Windows uses python22.dll instead of libpython2.2.so
PYVER_NODOT = $(subst .,,$(PYTHON_VERSION))

include $(TOP)/mk/os.make

ifeq ($(OS),Windows)
datadir = $(bindir)/share
endif

# Location for fetching third party binaries.
# Need to use curl --insecure because SSL_CERT_FILE is set below to non-existent file on Mac.
PREREQS_ARCHIVE = https://cxtoolshed.rbvi.ucsf.edu/prereqs
FETCH_PREREQ = curl --silent --show-error --fail --insecure -O
PREREQS_UPLOAD = plato.cgl.ucsf.edu:/usr/local/projects/chimerax/www/data/prereqs

# Location for large test data files
TEST_DATA_ARCHIVE = https://www.rbvi.ucsf.edu/chimerax/data/test_data
FETCH_TEST_DATA = curl --silent --show-error --insecure -O

ifeq ($(OS),Linux)
# need root CAs for https in our Python
ifneq (,$(wildcard /etc/ssl/certs/ca-certificates.crt))
	# Debian/Ubuntu/Gentoo etc.
	export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else ifneq (,$(wildcard /etc/pki/tls/certs/ca-bundle.crt))
	# Fedora/RHEL
	export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
else ifneq (,$(wildcard /etc/ssl/ca-bundle.pem))
	# OpenSUSE
	export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem
else ifneq (,$(wildcard /etc/pki/tls/cacert.pem))
	# OpenELEC
	export SSL_CERT_FILE=/etc/pki/tls/cacert.pem
endif
endif


APP_NAME = ChimeraX
ifdef FLATPAK_DIST
APP_FILENAME = /app
else
APP_FILENAME = $(APP_NAME).app
endif
CHIMERAX_APP = $(wildcard $(TOP)/ChimeraX*.app)
ifeq ($(OS),Windows)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX.exe
endif
ifeq ($(OS),Darwin)
CHIMERAX_EXE = $(CHIMERAX_APP)/Contents/bin/ChimeraX
endif
ifeq ($(OS),Linux)
CHIMERAX_EXE = $(CHIMERAX_APP)/bin/ChimeraX
endif


ifeq ($(OS),Darwin)
frameworkdir = $(build_prefix)/Library/Frameworks
app_prefix = $(TOP)/$(APP_FILENAME)/Contents
app_frameworkdir =  $(app_prefix)/Library/Frameworks
else
ifneq (,$(patsubst /%,,$(APP_FILENAME)))
app_prefix = $(TOP)/$(APP_FILENAME)
else
app_prefix = $(APP_FILENAME)
endif
endif
app_bindir = $(app_prefix)/bin
app_includedir = $(app_prefix)/include
app_libdir = $(app_prefix)/lib
ifdef WIN32
app_shlibdir = $(app_bindir)
else
app_shlibdir = $(app_libdir)
endif
ifeq ($(OS),Windows)
app_datadir = $(app_bindir)/share
else
app_datadir = $(app_prefix)/share
endif

ifndef WIN32
RSYNC = rsync -rltWv --executability
else
RSYNC = $(bindir)/rsync.convert -rlptWv
endif
ifdef FLATPAK_DIST
RSYNC := $(bindir)/$(RSYNC)
endif

ifdef WIN32
PYTHON_INCLUDE_DIRS = -I'$(shell cygpath -m '$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)')'
PYTHON_LIBRARY_DIR = $(bindir)/Lib
APP_PYTHON_LIBRARY_DIR = $(app_bindir)/Lib
PYTHON_BIN = $(bindir)/python.exe
APP_PYTHON_BIN = $(app_bindir)/python.exe
APP_EXE = $(app_bindir)/$(APP_NAME)-console.exe
CYTHON_EXE = $(bindir)/Scripts/cython.exe
else ifdef USE_MAC_FRAMEWORKS
PYTHON_INCLUDE_DIRS = $(shell $(bindir)/python$(PYTHON_VERSION)$(PYTHON_ABI)-config --includes)
PYTHON_FRAMEWORK = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)
APP_PYTHON_FRAMEWORK = $(app_frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)
APP_PYTHON_FRAMEWORK_LIBDIR = $(app_frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/lib
PYTHON_LIBRARY_DIR = $(libdir)/python$(PYTHON_VERSION)
APP_PYTHON_LIBRARY_DIR = $(app_libdir)/python$(PYTHON_VERSION)
PYTHON_BIN = $(bindir)/python$(PYTHON_VERSION)
APP_PYTHON_BIN = $(app_bindir)/python$(PYTHON_VERSION)
APP_EXE = $(app_bindir)/$(APP_NAME)
CYTHON_EXE = $(bindir)/cython
else
PYTHON_INCLUDE_DIRS = -I$(includedir)/python$(PYTHON_VERSION)$(PYTHON_ABI)
PYTHON_LIBRARY_DIR = $(libdir)/python$(PYTHON_VERSION)
APP_PYTHON_LIBRARY_DIR = $(app_libdir)/python$(PYTHON_VERSION)
PYTHON_BIN = $(bindir)/python$(PYTHON_VERSION)
APP_PYTHON_BIN = $(app_bindir)/python$(PYTHON_VERSION)
APP_EXE = $(app_bindir)/$(APP_NAME)
CYTHON_EXE = $(bindir)/cython
endif
PYTHON_EXE = $(PYTHON_BIN) -I
APP_PYTHON_EXE = $(APP_PYTHON_BIN) -I
PYSITEDIR = $(PYTHON_LIBRARY_DIR)/site-packages
APP_PYSITEDIR = $(APP_PYTHON_LIBRARY_DIR)/site-packages
APP_PIP = $(APP_EXE) -m pip
APP_BOOTSTRAP_PIP = $(APP_PYTHON_EXE) -m pip
ifeq ($(OSARCH),DarwinIntel64)
ifneq ($(MACOSV1),10)
# To get Python to install wheel filenames with os version 11_0, 12_0, ...
# Details in ChimeraX ticket #4923.
APP_BOOTSTRAP_PIP = env SYSTEM_VERSION_COMPAT=0 $(APP_PYTHON_EXE) -m pip
endif
endif

ifeq ($(OS),Darwin)
export SSL_CERT_FILE = $(PYSITEDIR)/certifi/cacert.pem
endif

PYLINT = $(PYTHON_EXE) -m flake8

# common makefile targets
.PHONY: all install app-install clean
