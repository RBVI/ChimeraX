TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/map
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/map

SUBDIRS	= _map data filter fit series

PYSRCS = __init__.py defaultsettings.py emdb_fetch.py grayscale.py \
	measure.py molmap.py mouselevel.py \
	moveplanes.py session.py slice.py solid.py volume.py volumecommand.py

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

lint:
	$(PYLINT) $(PYSRCS)
