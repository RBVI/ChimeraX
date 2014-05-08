TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/map

SUBDIRS	= data fit

PYSRCS = __init__.py defaultsettings.py grayscale.py molmap.py moveplanes.py \
	session.py slice.py solid.py subsample.py volume.py volumecommand.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(SUBDIR_INSTALL): $(PKG_DIR)

clean:
	rm -rf __pycache__
