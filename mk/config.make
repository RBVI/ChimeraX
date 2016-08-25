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

PYTHON_VERSION = 3.5
ifndef DEBUG
PYTHON_ABI = m
else
ifneq (,$(wildcard $(includedir)/python$(PYTHON_VERSION)dm))
PYTHON_ABI = dm
else
# didn't find debug include files, Python not compiled with debugging,
# so revert to regular Python ABI
PYTHON_ABI = m
endif
endif
# Windows uses python22.dll instead of libpython2.2.so
PYVER_NODOT = $(subst .,,$(PYTHON_VERSION))

include $(TOP)/mk/os.make

ifeq ($(OS),Windows)
datadir = $(bindir)/share
endif

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
