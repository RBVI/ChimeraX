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

LIBNAME = pdb
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))

$(LIBRARY): $(OBJS)
	$(LIB_LINK)

include:
	mkdir include

lib:
	mkdir lib

install: $(LIBRARY) include lib
	$(RSYNC) $(LIBRARY) lib
	$(RSYNC) $(wildcard *.h) include/$(DIRNAME)/
	#if [ -d "$(app_includedir)" ]; then \
	#	$(RSYNC) $(wildcard *.h) $(app_includedir)/$(DIRNAME)/; \
	#fi
ifdef WIN32
	#if [ -d "$(app_libdir)" ]; then \
	#	$(RSYNC) $(LIBRARY) $(app_libdir); \
	#fi
endif

clean:
	rm -rf $(LIBRARY) $(OBJS) lib include
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
endif
