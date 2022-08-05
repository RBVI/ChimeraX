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

PKG_DIR		= $(PYSITEDIR)/chimerax/seqalign/align_algs
ARRAYS_DIR	= $(APP_PYSITEDIR)/chimerax/arrays

PYMOD_NAME	= _nw
SRCS		= nw.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -I$(ARRAYS_DIR)/include
LIBS		+= $(PYTHON_LIB)
ifeq ($(WIN32),msvc)
LIBS		+= libalign_algs.lib /LIBPATH:$(ARRAYS_DIR)/lib libarrays.lib
else
LIBS		+= -lalign_algs -L$(ARRAYS_DIR)/lib -larrays
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
	rm -f $(PYMOD_NAME).exp $(PYMOD_NAME).lib
endif
