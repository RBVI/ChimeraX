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

TOP= ../../../../../..
include $(TOP)/mk/config.make

SUBDIRS	= idatm
include $(TOP)/mk/subdir.make

#LIBNAME = tmpl
DIRNAME = $(shell basename `pwd`)
DIRNAME := atomstruct/$(DIRNAME:_cpp=)

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
INCS += $(PYTHON_INCLUDE_DIRS) -I$(includedir) -I../../include

all: $(OBJS)

install: all
	-mkdir -p ../../include/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) ../../include/$(DIRNAME)

templates:
	cd leap-templates ; \
	for i in amino namino camino nucleic general ions; do \
		echo extracting $$i ; \
		./extract.csh $$i ; \
		mv $$i.cpp .. ; \
	done

clean:
	rm -f $(OBJS) $(SHLIB)
ifdef WIN32
	rm -f vc*.pdb
endif
