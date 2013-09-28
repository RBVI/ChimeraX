TOP = ../../..
include $(TOP)/mk/config.make

#SUBDIRS	= amira apbs ccp4 cmap delphi dock dsn6 emanhdf gaussian \
#	gopenmol imagestack imod macmolplt mrc netcdf pif priism \
#	profec situs spider tiff tom_em uhbd xplor

# TODO: Port other file readers from Python 2 to Python 3.

SUBDIRS	= ccp4 cmap dsn6 mrc

PKG_DIR = $(PYSITEDIR)/hydra/VolumeData

PYSRCS = __init__.py arraygrid.py arrays.py datacache.py fileformats.py \
	griddata.py memoryuse.py opendialog.py progress.py readarray.py \
	regions.py subsample.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
