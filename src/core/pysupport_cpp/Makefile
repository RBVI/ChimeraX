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

DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))

install:
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)/
	if [ -d "$(app_includedir)" ]; then \
		$(RSYNC) $(wildcard *.h) $(app_includedir)/$(DIRNAME)/; \
	fi

clean:
	rm -f $(OBJS)
