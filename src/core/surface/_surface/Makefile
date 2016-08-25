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

PKG_DIR = $(PYSITEDIR)/chimerax/core/surface
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/surface

PYMOD_NAME = _surface
SRCS	= border.cpp capper.cpp connected.cpp measure.cpp normals.cpp \
	  patches.cpp refinemesh.cpp sasa.cpp subdivide.cpp surface.cpp \
	  triangulate.cpp tube.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifdef WIN32
LIBS	+= libarrays.lib $(PYTHON_LIB) $(OPENGL_LIBS)
else
LIBS	+= -larrays $(PYTHON_LIB) $(OPENGL_LIBS)
endif

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PKG_DIR)
endif
endif

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
ifdef WIN32
	rm -f $(PYMOD_NAME).{exp,pdb} vc*.pdb
endif
