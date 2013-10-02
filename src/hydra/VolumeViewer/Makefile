TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/VolumeViewer

PYSRCS = __init__.py defaultsettings.py grayscale.py moveplanes.py \
	selectregion.py session.py slice.py solid.py subsample.py \
	volume.py volumecommand.py

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
