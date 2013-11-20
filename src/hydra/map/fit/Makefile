TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/FitMap

PYSRCS = __init__.py fitcmd.py fitlist.py fitmap.py move.py \
	search.py sequence.py session.py

__pycache__/%.cpython-33.pyc : %.py
	$(bindir)/python3 -t -m py_compile '$<'

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

clean:
	rm -rf __pycache__
