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

TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core

LIBNAME = arrays
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= pythonarray.cpp rcarray.cpp refcount.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= pythonarray.h rcarray.h refcount.h
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	+= $(PYTHON_LIB)

all: $(SHLIB)

install: all
	$(RSYNC) $(SHLIB) $(shlibdir)
ifdef WIN32
	$(RSYNC) $(LIBRARY) $(libdir)
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb $(shlibdir)
endif
endif
	$(RSYNC) imex.h $(HDRS) $(includedir)/$(DIRNAME)/
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) imex.h $(HDRS) $(app_includedir)/$(DIRNAME)/; \
	fi
ifdef WIN32
	if [ -d "$(app_libdir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
	fi
endif
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(SHLIB) $(OBJS)
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
	rm -f $(PYMOD_NAME).exp $(PYMOD_NAME).lib
endif
