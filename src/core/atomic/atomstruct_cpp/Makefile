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

SUBDIRS	= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifeq ($(OS),Darwin)
CXXFLAGS	+= -Wno-logical-op-parentheses
endif
LIBS	+= tmpl/*.$(OBJ_EXT)
ifeq ($(WIN32),msvc)
LIBS	+= libelement.lib libioutil.lib libchutil.lib libappdirs.lib liblogger.lib libarrays.lib $(PYTHON_LIB)
else
LIBS	+= -lelement -lioutil -lchutil -lappdirs -llogger -larrays $(PYTHON_LIB)
endif

$(SHLIB): imex.h $(OBJS) $(SUBDIR_INSTALL)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
ifdef WIN32
	$(RSYNC) $(LIBRARY) $(libdir)
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb $(shlibdir)
endif
endif
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)/
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) imex.h $(wildcard *.h) $(app_includedir)/$(DIRNAME)/; \
	fi
ifdef WIN32
	if [ -d "$(app_libdir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
	fi
endif
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	rm -f $(OBJS) $(SHLIB) imex.h
ifdef WIN32
	rm -f lib$(LIBNAME).{exp,pdb} vc*.pdb
endif

$(SUBDIR_INSTALL): imex.h

Atom.o: Atom.h Bond.h ChangeTracker.h CoordSet.h destruct.h Structure.h PBGroup.h \
	Pseudobond.h PythonInstance.h Residue.h
AtomicStructure.o: Atom.h AtomicStructure.h Bond.h CoordSet.h destruct.h Structure.h \
	PBGroup.h Pseudobond.h PythonInstance.h Residue.h seq_assoc.h
AtomTypes.o: AtomicStructure.h Bond.h Coord.h Residue.h tmpl/TAexcept.h
Bond.o: Atom.h Bond.h Chain.h Connection.h Structure.h PythonInstance.h Residue.h Sequence.h
Chain.o: Chain.h PythonInstance.h StructureSeq.h
ChangeTracker.o: ChangeTracker.h
CompSS.o: Atom.h AtomicStructure.h Coord.h Residue.h Structure.h
connect.o: Atom.h Bond.h connect.h Coord.h destruct.h Structure.h MolResId.h PBGroup.h \
	Residue.h string_types.h tmpl/Residue.h tmpl/Atom.h tmpl/residues.h
CoordSet.o: Atom.h CoordSet.h destruct.h PythonInstance.h Structure.h
destruct.o: destruct.h
Structure.o: Atom.h Bond.h ChangeTracker.h CoordSet.h destruct.h PBGroup.h PBManager.h \
	Pseudobond.h PythonInstance.h Structure.h Residue.h
StructureSeq.o: Structure.h Chain.h ChangeTracker.h destruct.h PythonInstance.h Structure.h Sequence.h Residue.h
PBGroup.o: Atom.h destruct.h Structure.h PBGroup.h Pseudobond.h PythonInstance.h
PBManager.o: destruct.h Structure.h PBGroup.h PBManager.h PythonInstance.h
Point.o: Point.h
Pseudobond.o: Atom.h ChangeTracker.h Connection.h PBGroup.h PBManager.h Pseudobond.h \
	PythonInstance.h
Residue.o: Atom.h Bond.h destruct.h PythonInstance.h Residue.h tmpl/TemplateCache.h
RibbonXSection.o: RibbonXSection.h
Ring.o: Atom.h Bond.h Coord.h PythonInstance.h Real.h Ring.h
RingCalc.o: Bond.h Structure.h Residue.h
seq_assoc.o: Chain.h Structure.h Residue.h seq_assoc.h
Sequence.o: Sequence.h
