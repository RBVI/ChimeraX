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
	rm -f lib$(LIBNAME).{pdb,exp} vc*.pdb
endif
