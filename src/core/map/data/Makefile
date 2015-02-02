TOP = ../../../..
include $(TOP)/mk/config.make

#SUBDIRS	= amira apbs ccp4 cmap delphi dock dsn6 emanhdf gaussian \
#	gopenmol imagestack imod macmolplt mrc netcdf pif priism \
#	profec situs spider tiff tom_em uhbd xplor

# TODO: Port other file readers from Python 2 to Python 3.

SUBDIRS	= ccp4 cmap dsn6 imagestack mrc

PKG_DIR = $(PYSITEDIR)/chimera/core/map/data
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/map/data

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
