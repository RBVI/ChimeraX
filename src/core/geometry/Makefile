TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/geometry
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/geometry

SUBDIRS	= _geometry

PYSRCS = bins.py bounds.py icosahedron.py matrix.py place.py sphere.py spline.py vector.py

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
	#if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(APP_PKG_DIR); fi

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	#if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

clean:
	rm -rf __pycache__

lint:
	$(PYLINT) $(PYSRCS)
