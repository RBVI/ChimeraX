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
include Makefile.common

all: $(XDRFILE_LIB)

install: all

$(XDRFILE_LIB): $(XDRFILE_LIB_SOURCE)

$(XDRFILE_LIB_SOURCE):
	# Use separate makefile for xdr so that PREREQ_MAKE can be turned on
	$(MAKE) -f Makefile.xdrfile

clean:
	rm -rf $(XDRFILE_LIB_SOURCE) $(BUILD)
