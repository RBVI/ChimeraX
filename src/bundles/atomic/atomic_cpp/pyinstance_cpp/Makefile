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

LIBNAME = pyinstance
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= $(PYTHON_INCLUDE_DIRS)
ifneq ($(OS),Windows)
CXXFLAGS	+= -Wno-logical-op-parentheses
endif
LIBS	+= $(PYTHON_LIB)

$(SHLIB): imex.h imex.map.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) ../lib
ifdef WIN32
	$(RSYNC) $(LIBRARY) ../lib
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb ../lib
endif
endif
	$(RSYNC) imex.h imex.map.h $(wildcard *.h) ../include/$(DIRNAME)/

PythonInstance.o: imex.map.h

imex.map.h: imex.h
	sed -e 's/$(UPLIBNAME)/$(UPLIBNAME)_MAP/' < imex.h > imex.map.h

clean:
	rm -f $(OBJS) $(SHLIB) imex.h imex.map.h
ifdef WIN32
	rm -f lib$(LIBNAME).{lib,exp,pdb} vc*.pdb
endif

PythonInstance.o: PythonInstance.declare.h PythonInstance.instantiate.h
