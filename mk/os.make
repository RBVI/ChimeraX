# -*- mode: makefile -*-
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

include $(TOP)/mk/detectOS.make

#### Linux

ifeq ($(OS),Linux)
	BIN_PLATFORM = linux_x86_64
	OBJ_EXT = o
	LIB_EXT = a
	LIB_LINK = ar rc $(LIBRARY) $(OBJS)
	RANLIB = ranlib
	SHLIB_EXT = so
	SHLIB_LINK = $(LOADER) -fuse-ld=gold -Wl,--no-allow-shlib-undefined,--as-needed $(LDFLAGS) -shared -o $(SHLIB) $(OBJS) $(LIBS)
	PROG_EXT =
	PROG_LINK = $(LOADER) -Wl,--no-undefined,--as-needed $(LDFLAGS) -o $(PROG) $(OBJS) $(LIBS)

	# Add Debian's dpkg-buildflags additions for hardening code to
	# CFLAGS, CXXFLAGS, CPPFLAGS (OPT), and LDFLAGS.
	HARDENING = 1
ifdef DEBUG
	OPT = -g -Wall -Wextra
else
	OPT = -O3 -Wall -Wextra
ifdef HARDENING
	OPT += -D_FORTIFY=2
endif
endif
	GCC_VER	= $(shell $(CC) -dumpversion)
	CC = gcc -pipe -fPIC -std=gnu99 -fdebug-prefix-map=$(build_prefix)=.
	CXX = g++ -pipe -fPIC -fdebug-prefix-map=$(build_prefix)=.
ifdef HARDENING
	CC += -fstack-protector-strong
	CXX += -fstack-protector-strong
	LDFLAGS = -Wl,-Bsymbolic-functions -Wl,-z,relro
endif
ifneq (,$(shell echo $(GCC_VER) | sed -e 's/^[1-3]\..*//' -e 's/^4\.[0-6]\..*//'))
	# gcc 4.7 or newer
	CXX += -std=c++11
else
	ERROR := $(error "gcc $(GCC_VER) is too old")
endif

ifndef PREREQ_MAKE
	# require explicit exporting in all code we write
	CC += -fvisibility=hidden
	CXX += -fvisibility-ms-compat
endif

	PYDEF = -DPyMODINIT_FUNC='extern "C" __attribute__((__visibility__("default"))) PyObject*'
	PYMOD_EXT = so
	PYMOD_LINK = $(LOADER) -fuse-ld=gold $(LDFLAGS) -shared -Wl,--no-allow-shlib-undefined -o $(PYMOD) $(OBJS) $(LIBS)
	PYTHON_LIB = -L$(libdir) -lpython$(PYTHON_VERSION)$(PYTHON_ABI)

	OPENGL_LIBS = -L$(libdir) -lGLU -lGL
	INCS = -I$(includedir)
	LIBS = -L$(libdir)
endif

#### Darwin, a.k.a., Apple Mac OS X

ifeq ($(OS),Darwin)
# Architecture is x86_64 or arm64
MACOS_ARCH = $(shell uname -m)
ifeq (,$(MACOSX_DEPLOYMENT_TARGET))
	ifeq ($(MACOSV1),10)
		# Major versions are 10.14, 10.15, 11, 12, ...
		WHEEL_MACOS_VER = $(subst .,_,$(OSXVER))
	else
		# Wheel names are 11_0 even for macOS 11.4 since the 4 is a patch version.
		WHEEL_MACOS_VER = $(MACOSV1)_0
	endif
	BIN_PLATFORM=macosx_$(WHEEL_MACOS_VER)_$(MACOS_ARCH)
else
	BIN_PLATFORM=macosx_$(subst .,_,$(MACOSX_DEPLOYMENT_TARGET))_$(MACOS_ARCH)
endif
	USE_MAC_FRAMEWORKS = 1
	# *nix system
	OBJ_EXT = o
	LIB_EXT = a
	LIB_LINK = ar crs $(LIBRARY) $(OBJS)
	RANLIB = ranlib
	SHLIB_EXT = dylib
	SHLIB_LINK = $(LOADER) $(LDFLAGS) -dynamiclib -headerpad_max_install_names -install_name @rpath/$(SHLIB) -o $(SHLIB) $(OBJS) $(LIBS)
	PROG_EXT =
	PROG_LINK = $(LOADER) $(LDFLAGS) -o $(PROG) $(OBJS) $(LIBS)
	# sysctl needs to be on path for numpy, numexpr, pytables
	export PATH := $(PATH):/usr/sbin
ifdef DEBUG
	OPT = -g -Wall -Wextra
else
	OPT = -O3 -Wall -Wextra
endif
	CC = clang -pipe
	CXX = clang++ -std=c++11 -stdlib=libc++ -pipe
	SDK_PATH = $(shell echo | gcc -xc -E -v - 2>&1 | grep '\.sdk/usr/include$$' | cut -d . -f 1,2,3,4).sdk
ifndef PREREQ_MAKE
	CC += -fvisibility=hidden
	CXX += -fvisibility-ms-compat
endif

	PYDEF = -DPyMODINIT_FUNC='extern "C" __attribute__((__visibility__("default"))) PyObject*'
ifdef USE_MAC_FRAMEWORKS
	PYTHON_LIB = -F$(frameworkdir) -framework Python
else
	PYTHON_LIB = -L$(libdir) -lpython$(PYTHON_VERSION)$(PYTHON_ABI)
endif
	PYMOD_EXT = so
	#PYMOD_LINK = $(LOADER) $(LDFLAGS) -bundle -bundle_loader `which python3` -o $(PYMOD) $(OBJS) $(LIBS) $(PYTHON_LIB)
	PYMOD_LINK = $(LOADER) $(LDFLAGS) -bundle -bundle_loader $(bindir)/python3 -o $(PYMOD) $(OBJS) $(LIBS) $(PYTHON_LIB)

	OPENGL_LIBS = -framework OpenGL
	INCS = -I$(includedir)
	LIBS = -L$(libdir)
endif

# Microsoft Windows

ifeq ($(OS),Windows)
	BIN_PLATFORM = win_amd64
	shlibdir = $(bindir)

	OBJ_EXT = obj
	LIB_EXT = lib
	LIB_LINK = lib /OUT:$(LIBRARY) $(OBJS)
	RANLIB = :
	SHLIB_EXT = dll
	PYMOD_EXT = pyd
	SHLIB_LINK = $(CXX) $(LDFLAGS) /LD /Fe$(SHLIB) $(OBJS) $(LIBS); if [ -e $(SHLIB).manifest ]; then mt -nologo -manifest $(SHLIB).manifest -outputresource:$(SHLIB)\;2 ; fi
	PROG_EXT = .exe
	PROG_LINK = $(CXX) $(LDFLAGS) /Fe$(PROG) $(OBJS) $(LIBS); if [ -e $(PROG).manifest ]; then mt -nologo -manifest $(PROG).manifest -outputresource:$(PROG)\;1 ; fi

ifdef DEBUG
	OPT = /Zi /Wall /W3
else
	OPT = /Ox /W2
endif
	CC = cl /nologo /EHa /GR /GF /MD
	# Suppress warning, 4251, about "needs to have dll-interface to be used by clients of class"
	# since we can't fix it for standard container classes
	CXX = $(CC) /Zc:inline,rvalueCast,strictStrings /wd4251

	PYDEF =
	PYTHON_LIB = python$(PYVER_NODOT).$(LIB_EXT)
	PYMOD_LINK = $(CXX) $(LDFLAGS) /LD /Fe$(PYMOD) $(OBJS) $(LIBS); if [ -e $(PYMOD).manifest ]; then mt -nologo -manifest $(PYMOD).manifest -outputresource:$(PYMOD)\;2 ; fi

	OPENGL_LIBS = glu32.lib opengl32.lib
	INCS = -I'$(shell cygpath -m '$(includedir)')'
	LIBS = /link /LIBPATH:'$(shell cygpath -m '$(libdir)')'

.SUFFIXES: .obj .rc

.cpp.obj:
	$(CXX) $(CXXFLAGS) /c $<

.c.obj:
	$(CC) $(CFLAGS) /c $<

.rc.obj:
	rc $(DEFS) /i . /Fo$@ $<
endif

CFLAGS = $(OPT) $(INCS) $(DEFS)
CXXFLAGS = $(OPT) $(INCS) $(DEFS)
LDFLAGS = $(OPT)
LOADER = $(CXX)

LIBRARY = lib$(LIBNAME).$(LIB_EXT)
SHLIB = lib$(LIBNAME).$(SHLIB_EXT)
PYMOD = $(PYMOD_NAME).$(PYMOD_EXT)
PROG = $(PROG_NAME)$(PROG_EXT)
UPLIBNAME = $(shell echo $(LIBNAME) | tr "[:lower:]" "[:upper:]" | tr '-' '_')
imex.h:
	sed -e 's/LIBNAME/$(UPLIBNAME)/' < $(includedir)/imex.i > imex.h

NUMPY_INC += -I"$(shell $(PYTHON_EXE) -c "import numpy; print(numpy.get_include())")"

PYOBJS = $(addprefix __pycache__/,$(addsuffix .cpython-$(PYVER_NODOT).pyc,$(basename $(PYSRCS))))

.SUFFIXES: .rst .html
.rst.html:
	rst2html --title=$* $^ $@

.SUFFIXES: .py .pyc .pyo
ifdef UNIX
__pycache__/%.cpython-$(PYVER_NODOT).pyc : %.py
	$(bindir)/python3 -t -m py_compile '$<'

.py.pyc:
	@rm -f $@
	python3 -t -m py_compile '$<'

.py.pyo:
	@rm -f $@
	python3 -t -OO -m py_compile '$<'
endif

ifdef WIN32
__pycache__/%.cpython-$(PYVER_NODOT).pyc : %.py
	$(bindir)/python -t -m py_compile '$(shell cygpath -m $<)'

.py.pyc:
	@rm -f $@
	python3 -t -m py_compile '$(shell cygpath -m $<)'

.py.pyo:
	@rm -f $@
	python3 -t -OO -m py_compile '$(shell cygpath -m $<)'
endif
