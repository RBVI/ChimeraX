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

TOP= ../../../../..
include $(TOP)/mk/config.make

SUBDIRS	= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
EXPANDED_INCS := $(INCS)
INCS	= -I../include $(EXPANDED_INCS) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifeq ($(OS),Darwin)
CXXFLAGS	+= -Wno-logical-op-parentheses
endif
LIBS	+= tmpl/*.$(OBJ_EXT)
ifeq ($(WIN32),msvc)
LIBS	+= /LIBPATH:../lib libelement.lib libpyinstance.lib libioutil.lib libchutil.lib libappdirs.lib liblogger.lib libarrays.lib $(PYTHON_LIB)
else
LIBS	+= -L../lib -lelement -lpyinstance -lioutil -lchutil -lappdirs -llogger -larrays $(PYTHON_LIB)
endif

$(SHLIB): imex.h $(OBJS) $(SUBDIR_INSTALL)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) ../lib
ifdef WIN32
	$(RSYNC) $(LIBRARY) ../lib
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb ../lib
endif
endif
	$(RSYNC) imex.h $(wildcard *.h) ../include/$(DIRNAME)/

clean:
	rm -f $(OBJS) $(SHLIB) imex.h
ifdef WIN32
	rm -f lib$(LIBNAME).{lib,exp,pdb} vc*.pdb
endif

$(SUBDIR_INSTALL): imex.h

Atom.o: Atom.h Bond.h ChangeTracker.h CoordSet.h destruct.h PBGroup.h Pseudobond.h Residue.h \
	session.h Structure.h ../element_cpp/Element.h ../pyinstance_cpp/PythonInstance.declare.h \
	../pyinstance_cpp/PythonInstance.instantiate.h
AtomicStructure.o: Atom.h AtomicStructure.h Bond.h CoordSet.h destruct.h Structure.h \
	PBGroup.h Pseudobond.h Residue.h seq_assoc.h
AtomTypes.o: AtomicStructure.h Bond.h Coord.h Residue.h tmpl/TAexcept.h
Bond.o: Atom.h Bond.h Chain.h ChangeTracker.h Connection.h destruct.h Structure.h Residue.h \
	Sequence.h session.h ../pyinstance_cpp/PythonInstance.declare.h \
	../pyinstance_cpp/PythonInstance.instantiate.h
Chain.o: Chain.h ChangeTracker.h StructureSeq.h ../pyinstance_cpp/PythonInstance.declare.h \
	../pyinstance_cpp/PythonInstance.instantiate.h
ChangeTracker.o: ChangeTracker.h
CompSS.o: Atom.h AtomicStructure.h Coord.h Residue.h Structure.h
CoordSet.o: Atom.h ChangeTracker.h CoordSet.h destruct.h session.h Structure.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
destruct.o: destruct.h
PBGroup.o: Atom.h ChangeTracker.h destruct.h PBGroup.h Pseudobond.h session.h Structure.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
PBManager.o: ChangeTracker.h destruct.h Structure.h PBGroup.h PBManager.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
Point.o: Point.h
Pseudobond.o: Atom.h ChangeTracker.h Connection.h PBGroup.h PBManager.h Pseudobond.h session.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
Residue.o: Atom.h Bond.h ChangeTracker.h destruct.h Residue.h session.h tmpl/TemplateCache.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
RibbonXSection.o: RibbonXSection.h
Ring.o: Atom.h Bond.h Coord.h Real.h Ring.h ../pyinstance_cpp/PythonInstance.declare.h \
	../pyinstance_cpp/PythonInstance.instantiate.h
RingCalc.o: Bond.h Structure.h Residue.h
seq_assoc.o: Chain.h Structure.h Residue.h seq_assoc.h
Sequence.o: Sequence.h session.h ../pyinstance_cpp/PythonInstance.declare.h \
	../pyinstance_cpp/PythonInstance.instantiate.h
Structure.o: Atom.h Bond.h ChangeTracker.h CoordSet.h destruct.h PBGroup.h PBManager.h \
	Pseudobond.h Sequence.h Structure.h Residue.h ../element_cpp/Element.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
StructureSeq.o: Structure.h Chain.h ChangeTracker.h destruct.h Structure.h Sequence.h Residue.h \
	../pyinstance_cpp/PythonInstance.declare.h ../pyinstance_cpp/PythonInstance.instantiate.h
