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

PKG_DIR		= $(PYSITEDIR)/chimerax/md_crds

XDRFILE_VERSION = 1.1
XDRFILE_MINOR_VERSION = b
XDRFILE_LIB_SOURCE = xdrfile-$(XDRFILE_VERSION)$(XDRFILE_MINOR_VERSION)
XDRFILE_ARCHIVE = xdrfile-$(XDRFILE_VERSION).tar.gz
XDRFILE_LIB = $(XDRFILE_LIB_SOURCE)/src/.libs/libxdrfile.a
PATCHES = patch

PYMOD_NAME	= _gromacs
SRCS		= gromacs.cpp
OBJS		= $(SRCS:.cpp=.$(OBJ_EXT))
DEFS		+= $(PYDEF)
INCS 		+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC) -I$(XDRFILE_LIB_SOURCE)/include
LIBS		+= $(PYTHON_LIB)
LIBS		+= $(XDRFILE_LIB)

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)
ifdef WIN32
ifdef DEBUG
	$(RSYNC) $(PYMOD_NAME).pdb $(PKG_DIR)
endif
endif

$(OBJS): $(XDRFILE_LIB)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

$(XDRFILE_LIB): $(XDRFILE_LIB_SOURCE)
	cd $(XDRFILE_LIB_SOURCE) ; $(MAKE)

$(XDRFILE_LIB_SOURCE):
	tar zxf $(XDRFILE_ARCHIVE)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		if [ -f $$p ] ; then \
			(cd $(XDRFILE_LIB_SOURCE) ; patch -p0) < $$p ; \
		fi ; \
	done
endif
	cd $(XDRFILE_LIB_SOURCE) ; env $(ENV_CONFIGURE) ./configure --enable-shared

clean:
	rm -rf $(PYMOD) $(OBJS) $(XDRFILE_LIB_SOURCE) $(BUILD)
ifdef WIN32
	rm -f $(PYMOD_NAME).exp $(PYMOD_NAME).lib
endif
