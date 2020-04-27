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

TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/graphics
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/graphics
ARRAYS_DIR = $(APP_PYSITEDIR)/chimerax/arrays

PYMOD_NAME = _graphics
SRCS	= blend_rgba.cpp count.cpp graphics.cpp \
	  linux_swap_interval.cpp mac_swap_interval.cpp \
	  mesh_edges.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -I$(ARRAYS_DIR)/include

ifdef WIN32
LIBS	+= /LIBPATH:$(ARRAYS_DIR)/lib libarrays.lib $(PYTHON_LIB)
else
ifeq ($(OS),Darwin)
LIBS	+= -L$(ARRAYS_DIR)/lib -larrays $(PYTHON_LIB) -framework OpenGL
else
ifeq ($(OS),Linux)
LIBS	+= -L$(ARRAYS_DIR)/lib -larrays $(PYTHON_LIB) -lGL
else
LIBS	+= -larrays $(PYTHON_LIB)
endif
endif
endif

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) ../src
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb ../src
endif
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
ifdef WIN32
	rm -f $(PYMOD_NAME).{lib,exp,pdb} vc*.pdb
endif
