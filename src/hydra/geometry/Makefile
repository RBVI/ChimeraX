TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/geometry

PYSRCS = bins.py bounds.py icosahedron.py matrix.py place.py spline.py vector.py

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
