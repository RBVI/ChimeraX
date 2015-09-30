include $(TOP)/mk/detectOS.make

#### Linux

ifeq ($(OS),Linux)
	OBJ_EXT = o
	LIB_EXT = a
	LIB_LINK = ar rc $(LIBRARY) $(OBJS)
	RANLIB = ranlib
	SHLIB_EXT = so
	SHLIB_LINK = $(LOADER) $(LDFLAGS) -shared -o $(SHLIB) $(OBJS) $(LIBS)
	PROG_EXT =
	PROG_LINK = $(LOADER) $(LDFLAGS) -o $(PROG) $(OBJS) $(LIBS)

ifdef DEBUG
	OPT = -g -Wall -Wextra
else
	OPT = -O3 -Wall -Wextra
endif
	GCC_VER	= $(shell $(CC) -dumpversion)
	CC = gcc -pipe -fPIC -std=gnu99
	CXX = g++ -pipe -fPIC
ifneq (,$(shell echo $(GCC_VER) | sed -e 's/^[1-3]\..*//' -e 's/^4\.[0-6]\..*//'))
	# gcc 4.7 or newer
	CXX += -std=c++11
else
	ERROR := $(error "gcc $(GCC_VER) is too old")
endif
#TODO
#ifndef PREREQ_MAKE
#	# require explicit exporting in all code we write
#	CC += -fvisibility-ms-compat
#	CXX += -fvisibility-ms-compat
#endif

	PYDEF = -fvisibility-ms-compat -DPyMODINIT_FUNC='extern "C" __attribute__((__visibility__("default"))) PyObject*'
	PYMOD_EXT = so
	PYMOD_LINK = $(LOADER) $(LDFLAGS) -shared -o $(PYMOD) $(OBJS) $(LIBS)
	PYTHON_LIB = -L$(libdir) -lpython$(PYTHON_VERSION)$(PYTHON_ABI)

	OPENGL_LIBS = -L$(libdir) -lGLEW -lGL
endif

#### Darwin, a.k.a., Apple Mac OS X

ifeq ($(OS),Darwin)
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

	# SDK is one of the sdk arguments listed in `xcodebuild -showsdks`
	# SYSROOT is the path the the SDKs
	XCODE_SDKS = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
ifneq (,$(MACOSX_DEPLOYMENT_TARGET))
	SDK = macosx$(MACOSX_DEPLOYMENT_TARGET)
	ifneq (,$(wildcard $(XCODE_SDKS)))
		SYSROOT = $(XCODE_SDKS)/MacOSX$(MACOSX_DEPLOYMENT_TARGET).sdk
	else
        $(error unable to find SYSROOT for $(MACOSX_DEPLOYMENT_TARGET))
	endif
else ifneq (,$(wildcard $(XCODE_SDKS)/MacOSX10.9.sdk))
	export MACOSX_DEPLOYMENT_TARGET=10.9
	SYSROOT = $(XCODE_SDKS)/MacOSX10.9.sdk
	SDK = macosx10.9
else ifneq (,$(wildcard $(XCODE_SDKS)/MacOSX10.8.sdk))
	export MACOSX_DEPLOYMENT_TARGET=10.8
	SYSROOT = $(XCODE_SDKS)/MacOSX10.8.sdk
	SDK = macosx10.8
else ifneq (,$(wildcard $(XCODE_SDKS)/MacOSX10.7.sdk))
	export MACOSX_DEPLOYMENT_TARGET=10.7
	SYSROOT = $(XCODE_SDKS)/MacOSX10.7.sdk
	SDK = macosx10.7
else ifneq (,$(wildcard $(XCODE_SDKS)/MacOSX10.6.sdk))
	export MACOSX_DEPLOYMENT_TARGET=10.6
	SYSROOT = $(XCODE_SDKS)/MacOSX10.6.sdk
	SDK = macosx10.6
else
    $(error Unable to find Xcode sysroot)
endif
	# sysctl needs to be on path for numpy, numexpr, pytables
	export PATH := $(PATH):/usr/sbin
ifdef DEBUG
	OPT = -g -Wall -Wextra
else
	OPT = -O3 -Wall -Wextra
endif
	CC = clang --sysroot $(SYSROOT)
	CXX = clang++ --sysroot $(SYSROOT) -std=c++11 -stdlib=libc++
	EXTRA_CFLAGS = -fPIC
	EXTRA_CXXFLAGS = -fPIC -fvisibility-ms-compat

	PYDEF = -DPyMODINIT_FUNC='extern "C" __attribute__((__visibility__("default"))) PyObject*'
ifdef USE_MAC_FRAMEWORKS
	PYTHON_LIB = -F$(frameworkdir) -framework Python
else
	PYTHON_LIB = -L$(libdir) -lpython$(PYTHON_VERSION)$(PYTHON_ABI)
endif
	PYMOD_EXT = so
	#PYMOD_LINK = $(LOADER) $(LDFLAGS) -bundle -bundle_loader `which python3` -o $(PYMOD) $(OBJS) $(LIBS) $(PYTHON_LIB)
	PYMOD_LINK = $(LOADER) $(LDFLAGS) -bundle -bundle_loader $(bindir)/python3 -o $(PYMOD) $(OBJS) $(LIBS) $(PYTHON_LIB)

	OPENGL_LIBS = -L$(libdir) -lGLEW -framework OpenGL
endif

# Microsoft Windows

ifeq ($(OS),Windows)
	shlibdir = $(bindir)
	app_shlibdir = $(app_bindir)

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
	CC = cl /nologo /Zc:forScope /EHa /GR /GF /MD
	CXX = $(CC)

	PYDEF =
	PYTHON_LIB = python$(PYVER_NODOT).$(LIB_EXT)
	PYMOD_LINK = $(CXX) $(LDFLAGS) /LD /Fe$(PYMOD) $(OBJS) $(LIBS); if [ -e $(PYMOD).manifest ]; then mt -nologo -manifest $(PYMOD).manifest -outputresource:$(PYMOD)\;2 ; fi

	OPENGL_LIBS = glew32.lib opengl32.lib

.SUFFIXES: .obj

.cpp.obj:
	$(CXX) $(CXXFLAGS) /c $<

.c.obj:
	$(CC) $(CFLAGS) /c $<
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

NUMPY_INC += -I"$(shell $(bindir)/python$(PYTHON_VERSION) -c "import numpy; print(numpy.get_include())")"

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
.py.pyc:
	@rm -f $@
	python3 -t -m py_compile '$(shell cygpath -m $<)'

.py.pyo:
	@rm -f $@
	python3 -t -OO -m py_compile '$(shell cygpath -m $<)'
endif

