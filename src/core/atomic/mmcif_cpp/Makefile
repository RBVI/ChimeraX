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

PATCHES = $(wildcard *.patch)
WRAPPY = $(bindir)/wrappy3
ifeq (64,$(patsubst %64,64,$(OSARCH)))
WRAPPY	+= -L
ifeq ($(OSARCH),Windows64)
# size_t is unsigned long long on Windows64 
WRAPPY	+= -L
endif
endif

PATCHES = $(wildcard *.patch)

HDRS = mmcif.h
SRCS = mmcif.cpp template.cpp
GENSRCS += _mmcif.cpp
GENHDRS += $(GENSRCS:.cpp=.h)
OBJS = $(SRCS:.cpp=.$(OBJ_EXT)) $(GENSRCS:.cpp=.$(OBJ_EXT))
PYMOD_NAME = _mmcif

DEFS += $(PYDEF)
INCS += $(PYTHON_INCLUDE_DIRS)
ifeq ($(WIN32),msvc)
LIBS += libatomstruct.lib libelement.lib libreadcif.lib libwrappy3.lib liblogger.lib libarrays.lib libappdirs.lib $(PYTHON_LIB)
else
LIBS += -latomstruct -lelement -lreadcif -lwrappy3 -llogger -larrays -lappdirs $(PYTHON_LIB)
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
	rm -f $(PYMOD) $(OBJS) $(GENSRCS) $(GENHDRS) wrappy *.orig *.rej
ifeq ($(WIN32),msvc)
	rm -f $(PYMOD).manifest $(PYMOD_NAME).{lib,exp,ilk,pdb} vc*.pdb
endif

$(GENSRCS) $(OBJS): wrappy

wrappy: $(HDRS)
	$(WRAPPY) -d -N mmcif -m _mmcif $(HDRS)
ifneq (,$(PATCHES))
	for i in $(PATCHES) ; do \
		patch -b < $$i ; \
	done
endif
	touch wrappy

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
