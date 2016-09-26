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

TOP= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core

PYMOD_NAME = _appdirs
SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS)
ifeq ($(WIN32),msvc)
LIBS += libappdirs.lib
else
LIBS += -lappdirs
endif


$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PKG_DIR)
endif
endif

clean:
	rm -f $(OBJS) $(PYMOD)
ifdef WIN32
	rm -f $(PYMOD_NAME).pdb vc*.pdb
	rm -f $(PYMOD_NAME).exp $(PYMOD_NAME).lib
endif
