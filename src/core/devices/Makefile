TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/devices
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/devices

PYSRCS = __init__.py

SUBDIRS	= oculus spacenavigator

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

$(SUBDIR_INSTALL): dir

clean:
	rm -rf __pycache__

lint:
	$(PYLINT) $(PYSRCS)
