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

PKG_DIR = $(PYSITEDIR)/chimerax/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core

PYMOD_NAME = _mac_util
SRCS	= module.cpp setfileicon.cpp enablemultitouch.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT)) setfileicon_mac.o enablemultitouch_mac.o
DEFS	+= $(PYDEF)
INCS	+= $(PYTHON_INCLUDE_DIRS)
LIBS	+= $(PYTHON_LIB)
LIBS	+= -framework AppKit

all: $(PYMOD)

build install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
