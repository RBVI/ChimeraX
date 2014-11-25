TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/surface

PYSRCS = __init__.py area.py collada.py gridsurf.py read_stl.py \
	 sasa.py shapes.py split.py tube.py write_json.py write_stl.py

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
