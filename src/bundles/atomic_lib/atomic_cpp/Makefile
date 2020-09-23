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

SUBDIRS	+= atomstruct_cpp element_cpp pyinstance_cpp

include $(TOP)/mk/subdir.make

ifdef WIN32
ARRAYS_DIR = '$(shell cygpath -m '$(APP_PYSITEDIR)/chimerax/arrays')'
else
ARRAYS_DIR = $(APP_PYSITEDIR)/chimerax/arrays
endif

PYMOD_NAME = _load_libs
SRCS	= load_libs.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -Iinclude -I$(ARRAYS_DIR)/include
ifeq ($(WIN32),msvc)
LIBS	+= /LIBPATH:lib libatomstruct.lib libelement.lib libpyinstance.lib /LIBPATH:$(ARRAYS_DIR)/lib libarrays.lib liblogger.lib libappdirs.lib $(PYTHON_LIB)
else
LIBS	+= -Llib -latomstruct -lelement -lpyinstance -L$(ARRAYS_DIR)/lib -larrays -llogger -lappdirs $(PYTHON_LIB)
endif

all: $(PYMOD)

lib:
	mkdir lib

include:
	mkdir include
	mkdir include/atomic

data:
	mkdir data

install: all
	cat /dev/null > lib/__init__.py
	mv $(PYMOD) lib
	$(RSYNC) data include lib ../src

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
	install_name_tool -change @rpath/libatomstruct.dylib @loader_path/libatomstruct.dylib $(PYMOD)
	install_name_tool -change @rpath/libelement.dylib @loader_path/libelement.dylib $(PYMOD)
	install_name_tool -change @rpath/libpyinstance.dylib @loader_path/libpyinstance.dylib $(PYMOD)
endif

clean:
	rm -rf __pycache__ $(OBJS) lib include data build
ifdef WIN32
	rm -f vc*.pdb
endif

atomstruct_cpp.install: element_cpp.install pyinstance_cpp.install
element_cpp.install: pyinstance_cpp.install
pyinstance_cpp.install: lib include data
$(OBJS): atomstruct_cpp.install
