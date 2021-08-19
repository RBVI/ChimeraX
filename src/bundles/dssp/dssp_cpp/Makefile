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

PKG_DIR		= $(PYSITEDIR)/chimerax/core/atomic
APP_PKG_DIR		= $(PYSITEDIR)/chimerax/core/atomic
ifdef WIN32
PKG_DIR := $(shell cygpath -m '$(PKG_DIR)' | sed 's,/,\\\\\\\\,g')
APP_PKG_DIR := $(shell cygpath -m '$(APP_PKG_DIR)' | sed 's,/,\\\\\\\\,g')
endif

PYMOD_NAME	= _dssp
SRCS		= dssp.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS)
ifdef WIN32
LIBS		+= $(PYTHON_LIB) libatomstruct.lib libelement.lib libpyinstance.lib libarrays.lib liblogger.lib
else
LIBS		+= $(PYTHON_LIB) -latomstruct -lelement -lpyinstance -larrays -llogger
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
	rm -f lib$(LIBNAME).pdb vc*.pdb
	rm -f $(PYMOD_NAME).exp $(PYMOD_NAME).lib
endif
