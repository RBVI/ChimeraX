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

TOP = ../../../..
include $(TOP)/mk/config.make

SUBDIRS	+= ribbon_cpp

include $(TOP)/mk/subdir.make

ifdef WIN32
ARRAYS_DIR = '$(shell cygpath -m '$(APP_PYSITEDIR)/chimerax/arrays')'
ATOMIC_LIB_DIR = '$(shell cygpath -m '$(APP_PYSITEDIR)/chimerax/atomic_lib')'
else
ARRAYS_DIR = $(APP_PYSITEDIR)/chimerax/arrays
ATOMIC_LIB_DIR = $(APP_PYSITEDIR)/chimerax/atomic_lib
endif

LIBNAME = molc

SRCS	= molc.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= 
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -Iinclude -I$(ARRAYS_DIR)/include -I$(ATOMIC_LIB_DIR)/include
ifeq ($(WIN32),msvc)
LIBS	+= /LIBPATH:lib libatomstruct.lib libelement.lib libpyinstance.lib /LIBPATH:$(ARRAYS_DIR)/lib /LIBPATH:$(ATOMIC_LIB_DIR)/lib libarrays.lib liblogger.lib libappdirs.lib $(PYTHON_LIB)
else
LIBS	+= -Llib -L$(ATOMIC_LIB_DIR)/lib -latomstruct -lelement -lpyinstance -L$(ARRAYS_DIR)/lib -larrays -llogger -lappdirs $(PYTHON_LIB)
ifeq ($(OS),Linux)
LDFLAGS += -Wl,-rpath,\$${ORIGIN}/lib:\$${ORIGIN}/../../../.. -Wl,--as-needed
endif
endif

all:
	$(MAKE) install

cymol.$(PYMOD_EXT): $(SHLIB) cymol.pyx
	LDFLAGS="-L." $(PYTHON_EXE) setup-cymol.py build_ext -i
	mv chimerax/atomic/cymol.cp*.$(PYMOD_EXT) cymol.$(PYMOD_EXT)
	rmdir -p chimerax/atomic

cytmpl.$(PYMOD_EXT): $(SHLIB) cytmpl.pyx cymol.$(PYMOD_EXT)
	LDFLAGS="-L." $(PYTHON_EXE) setup-cytmpl.py build_ext -i
	mv chimerax/atomic/cytmpl.cp*.$(PYMOD_EXT) cytmpl.$(PYMOD_EXT)
	rmdir -p chimerax/atomic

install: $(SHLIB) cymol.$(PYMOD_EXT) cytmpl.$(PYMOD_EXT)
	$(RSYNC) $(SHLIB) cymol.$(PYMOD_EXT) cymol.pyx cytmpl.$(PYMOD_EXT) cytmpl.pyx ../src

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)
ifeq ($(OS),Darwin)
	#install_name_tool -change @rpath/libatomstruct.dylib @loader_path/lib/libatomstruct.dylib $(SHLIB)
	#install_name_tool -change @rpath/libelement.dylib @loader_path/lib/libelement.dylib $(SHLIB)
	#install_name_tool -change @rpath/libpyinstance.dylib @loader_path/lib/libpyinstance.dylib $(SHLIB)
endif

clean:
	rm -rf __pycache__ $(OBJS) $(SHLIB) \
		cymol.cpp cymol.$(PYMOD_EXT) \
		cytmpl.cpp cytmpl.$(PYMOD_EXT) build
ifdef WIN32
	rm -f lib$(LIBNAME).{lib,exp,pdb} vc*.pdb
endif

pylint:
	$(PYLINT) $(PYSRCS)

molobject.py: cymol.$(PYMOD_EXT)
