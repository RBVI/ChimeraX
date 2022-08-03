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

LIBNAME = chutil
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))

# LIBRARY is defined in os.make and evaluates to 
# lib$(LIBNAME).$(LIB_EXT)
$(LIBRARY): $(OBJS)
	# ar rc $(LIBRARY) $(OBJS)
	$(LIB_LINK)

build install: $(LIBRARY)
	# rsync the built library library to $(TOP)/build/lib
	$(RSYNC) $(LIBRARY) $(libdir)
	# rsync the headers to $(TOP)/build/include/chutil
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)/
	# if ChimeraX.app exists then put the headers in 
	# ChimeraX.app/include/chutil
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) $(wildcard *.h) $(app_includedir)/$(DIRNAME)/; \
	fi
	# if ChimeraX.app exists then put the library in ChimeraX.app/lib
	if [ -d "$(app_libdir)" ]; then \
		$(RSYNC) $(LIBRARY) $(app_libdir); \
	fi

clean:
	rm -f $(OBJS) $(LIBRARY)
ifdef WIN32
	rm -f lib$(LIBNAME).pdb vc*.pdb
endif
