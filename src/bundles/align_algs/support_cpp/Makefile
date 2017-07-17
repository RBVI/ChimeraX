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

LIBNAME = align_algs
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:support_cpp=align_algs)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= $(PYTHON_INCLUDE_DIRS)
LIBS	+= $(PYTHON_LIB)

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
		mkdir -p $(app_includedir)/$(DIRNAME); \
		$(RSYNC) $(wildcard *.h) $(app_includedir)/$(DIRNAME); \
	fi

clean:
	rm -f $(OBJS) $(LIBRARY)
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
endif
