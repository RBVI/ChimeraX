TOP= ../../../..
include $(TOP)/mk/config.make

SUBDIRS	= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS)
ifeq ($(OS),Darwin)
CXXFLAGS	+= -Wno-logical-op-parentheses
endif
LIBS	+= -L$(libdir) -lelement -ltmpl -llogger -lbasegeom -larrays $(PYTHON_LIB)

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

Atom.o: $(includedir)/basegeom/Coord.h $(includedir)/basegeom/Point.h \
	$(includedir)/basegeom/Sphere.h $(includedir)/basegeom/Connectible.h
AtomTypes.o: $(includedir)/basegeom/Coord.h
AtomicStructure.o: $(includedir)/basegeom/Graph.h \
	$(includedir)/basegeom/Graph.tcc $(includedir)/basegeom/destruct.h
Bond.o: $(includedir)/basegeom/Connection.h
Chain.o: $(includedir)/basegeom/destruct.h
CoordSet.o: $(includedir)/basegeom/Coord.h $(includedir)/basegeom/destruct.h
Pseudobond.o: $(includedir)/basegeom/Connection.h $(includedir)/basegeom/destruct.h
Residue.o: $(includedir)/basegeom/Rgba.h $(includedir)/basegeom/destruct.h
Ring.o: $(includedir)/basegeom/Coord.h $(includedir)/basegeom/Real.h
connect.o: $(includedir)/basegeom/Connectible.tcc \
	$(includedir)/basegeom/Graph.tcc $(includedir)/basegeom/destruct.h
