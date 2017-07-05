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

TOP= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/atomic
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/atomic

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = pdbio
DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS)
ifdef WIN32
LIBS += libatomstruct.lib libelement.lib liblogger.lib libpdb.lib libarrays.lib liblogger.lib libappdirs.lib $(PYTHON_LIB)
else
LIBS += -latomstruct -lelement -llogger -lpdb -larrays -llogger -lappdirs $(PYTHON_LIB)
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
	rm -f $(PYMOD_NAME).{exp,lib,pdb} vc*.pdb
endif

PDBio.o: $(includedir)/pdb/PDB.h $(includedir)/atomstruct/Atom.h \
	$(includedir)/atomstruct/AtomicStructure.h $(includedir)/atomstruct/Bond.h \
	$(includedir)/atomstruct/connect.h $(includedir)/atomstruct/CoordSet.h \
	$(includedir)/atomstruct/destruct.h $(includedir)/atomstruct/Residue.h \
	$(includedir)/atomstruct/Sequence.h $(includedir)/atomstruct/Structure.h \
	$(includedir)/logger/logger.h
