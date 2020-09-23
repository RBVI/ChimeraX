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

TOP = ../../../..
include $(TOP)/mk/config.make

SUBDIRS	+= atomstruct_cpp element_cpp pyinstance_cpp

include $(TOP)/mk/subdir.make

all: lib data include

lib:
	mkdir lib

include:
	mkdir include
	mkdir include/atomic

data:
	mkdir data

install: all
	$(RSYNC) data include lib ../src

clean:
	rm -rf __pycache__ $(OBJS) lib include data build
ifdef WIN32
	rm -f vc*.pdb
endif

atomstruct_cpp.install: element_cpp.install pyinstance_cpp.install
element_cpp.install: pyinstance_cpp.install
pyinstance_cpp.install: lib include data
