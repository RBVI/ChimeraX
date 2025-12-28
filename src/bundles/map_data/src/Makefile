# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = ../../../..
include $(TOP)/mk/config.make

SUBDIRS	= amira apbs brix ccp4 cmap delphi deltavision dock dsn6 emanhdf gaussian \
	  gopenmol hdf imagestack imagic imod ims lammps macmolplt mrc priism profec \
	  pif situs spider tom_em uhbd xplor

PKG_DIR = $(PYSITEDIR)/chimerax/map_data
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/map_data

PYSRCS = __init__.py arraygrid.py arrays.py datacache.py fileformats.py \
	griddata.py memoryuse.py opendialog.py progress.py readarray.py \
	regions.py subsample.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

clean:
	rm -rf __pycache__
