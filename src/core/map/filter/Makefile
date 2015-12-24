TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/map/filter

PYSRCS = __init__.py bin.py boxes.py cover.py falloff.py flatten.py flip.py fourier.py \
	gaussian.py laplace.py localcorr.py median.py permute.py ridges.py scale.py \
	square.py stack.py threshold.py tile.py unbend.py unroll.py vopcommand.py

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
