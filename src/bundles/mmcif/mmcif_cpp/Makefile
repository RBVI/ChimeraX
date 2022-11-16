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
ARRAYS_DIR = $(APP_PYSITEDIR)/chimerax/arrays

HDRS = mmcif.h _mmcif.h
SRCS = mmcif.cpp template.cpp corecif.cpp _mmcif.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _mmcif

DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS) -I$(ARRAYS_DIR)/include
ifeq ($(WIN32),msvc)
LIBS += /LIBPATH:../lib libatomstruct.lib libelement.lib libpyinstance.lib libreadcif.lib liblogger.lib /LIBPATH:$(ARRAYS_DIR)/lib libarrays.lib $(PYTHON_LIB)
else
LIBS += -Llib -latomstruct -lelement -lpyinstance -lreadcif -llogger -L$(ARRAYS_DIR)/lib -larrays $(PYTHON_LIB)
endif

lib:
	mkdir lib

include:
	mkdir include

source: lib include
	cd readcif_cpp; make install

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
	cd readcif_cpp; make clean
	rm -rf $(PYMOD) $(OBJS) include lib
ifeq ($(WIN32),msvc)
	rm -f $(PYMOD).manifest $(PYMOD_NAME).{lib,exp,ilk,pdb} vc*.pdb
endif

mmcif.o: $(includedir)/atomstruct/Atom.h \
	$(includedir)/atomstruct/AtomicStructure.h $(includedir)/atomstruct/Bond.h \
	$(includedir)/atomstruct/CoordSet.h $(includedir)/atomstruct/connect.h \
	$(includedir)/atomstruct/PBGroup.h $(includedir)/atomstruct/Pseudobond.h \
	$(includedir)/atomstruct/Residue.h $(includedir)/atomstruct/Structure.h \
	$(includedir)/atomstruct/tmpl/Atom.h $(includedir)/atomstruct/tmpl/Residue.h \
	$(includedir)/atomstruct/string_types.h $(includedir)/logger/logger.h
template.o: $(includedir)/atomstruct/Atom.h $(includedir)/atomstruct/AtomicStructure.h \
	$(includedir)/atomstruct/Bond.h $(includedir)/atomstruct/connect.h \
	$(includedir)/atomstruct/CoordSet.h \
	$(includedir)/atomstruct/Residue.h $(includedir)/atomstruct/Sequence.h
