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

PKG_DIR = $(PYSITEDIR)/chimerax/core/map

PYMOD_NAME = _map
SRCS	= colors.cpp combine.cpp contourdata.cpp contourpy.cpp \
	  distgrid.cpp extendmap.cpp fitting.cpp fittingpy.cpp  \
	  gaussian.cpp  histogram.cpp interpolate.cpp interpolatepy.cpp \
	  localcorr.cpp module.cpp moments.cpp occupancy.cpp squaremesh.cpp transfer.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifdef WIN32
LIBS	+= libarrays.lib $(PYTHON_LIB)
else
LIBS	+= -larrays $(PYTHON_LIB)
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
