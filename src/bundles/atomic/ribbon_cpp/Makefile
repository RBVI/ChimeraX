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

TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/atomic
INSTALL_MOD = ../../src

ifdef WIN32
ARRAYS_DIR = '$(shell cygpath -m '$(APP_PYSITEDIR)/chimerax/arrays')'
ATOMIC_LIB_DIR = '$(shell cygpath -m '$(APP_PYSITEDIR)/chimerax/atomic_lib')'
else
ARRAYS_DIR = $(APP_PYSITEDIR)/chimerax/arrays
ATOMIC_LIB_DIR = $(APP_PYSITEDIR)/chimerax/atomic_lib
endif

PYMOD_NAME = _ribbons
SRCS	= normals.cpp parse.cpp ribbons.cpp spline.cpp xsection.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS	+= $(PYDEF)
INCS	+= -I$(ATOMIC_LIB_DIR)/include $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -I$(ARRAYS_DIR)/include
#OPT = -g -Wall -Wextra

ifdef WIN32
LIBS	+= /LIBPATH:$(ATOMIC_LIB_DIR)/lib libatomstruct.lib /LIBPATH:$(ARRAYS_DIR)/lib libarrays.lib $(PYTHON_LIB)
else
LIBS	+= -L$(ATOMIC_LIB_DIR)/lib -latomstruct -L$(ARRAYS_DIR)/lib -larrays $(PYTHON_LIB)
endif

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(INSTALL_MOD)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(INSTALL_MOD)
endif
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
ifdef WIN32
	rm -f $(PYMOD_NAME).{lib,exp,pdb} vc*.pdb
endif

ribbons.o: normals.h spline.h xsection.h
normals.o: normals.h
spline.o: spline.h normals.h
xsection.o: xsection.h
