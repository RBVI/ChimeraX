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

LIBNAME = logger
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= $(PYTHON_INCLUDE_DIRS)
ifneq ($(OS),Windows)
CXX	+= -Wno-logical-op-parentheses
endif
LIBS	+= $(PYTHON_LIB)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

$(LIBRARY): imex.h $(OBJS)
	$(LIB_LINK)

build install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)/
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) imex.h $(wildcard *.h) $(app_includedir)/$(DIRNAME)/; \
	fi
	if [ -d "$(app_libdir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
	fi

clean:
	rm -f $(OBJS) $(LIBRARY) imex.h
ifdef WIN32
	rm -f lib$(LIBNAME).{lib,pdb,exp} vc*.pdb
endif
