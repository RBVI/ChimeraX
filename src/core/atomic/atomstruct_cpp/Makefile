TOP= ../../../..
include $(TOP)/mk/config.make

SUBDIRS	= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
ifeq ($(OS),Darwin)
CXXFLAGS	+= -Wno-logical-op-parentheses
endif
LIBS	+= -L$(libdir) -lelement -ltmpl -llogger -larrays $(PYTHON_LIB)

$(SHLIB): imex.h $(OBJS) $(SUBDIR_INSTALL)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h

$(SUBDIR_INSTALL): imex.h

Atom.o: Atom.h AtomicStructure.h Bond.h ChangeTracker.h CoordSet.h destruct.h PBGroup.h \
	Pseudobond.h Residue.h
AtomTypes.o: AtomTypes.h Bond.h Coord.h Residue.h tmpl/TAexcept.h
AtomicStructure.o: Atom.h AtomicStructure.h Bond.h CoordSet.h destruct.h PBGroup.h Pseudobond.h \
	Residue.h seq_assoc.h
Bond.o: Atom.h Bond.h Chain.h Residue.h Sequence.h
Chain.o: AtomicStructure.h Chain.h ChangeTracker.h destruct.h Sequence.h Residue.h
ChangeTracker.o: ChangeTracker.h
CoordSet.o: Atom.h AtomicStructure.h CoordSet.h destruct.h
Graph.o: Atom.h Bond.h destruct.h Graph.h
PBGroup.o: Atom.h AtomicStructure.h destruct.h PBGroup.h Pseudobond.h
PBManager.o: AtomicStructure.h destruct.h PBGroup.h
Point.o: Point.h
Pseudobond.o: Atom.h ChangeTracker.h PBGroup.h PBManager.h Pseudobond.h
Residue.o: Atom.h Bond.h destruct.h Residue.h tmpl/TemplateCache.h
RibbonXSection.o: RibbonXSection.h
Ring.o: Atom.h Bond.h Coord.h Real.h Ring.h
RingCalc.o: AtomicStructure.h Bond.h Residue.h
Sequence.o: Sequence.h
connect.o: Atom.h AtomicStructure.h Bond.h connect.h Coord.h destruct.h MolResId.h PBGroup.h \
	Residue.h string_types.h tmpl/Residue.h tmpl/Atom.h tmpl/residues.h
destruct.o: destruct.h
seq_assoc.o: AtomicStructure.h Chain.h Residue.h seq_assoc.h
