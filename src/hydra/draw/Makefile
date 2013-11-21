TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/draw

PYSRCS = __init__.py drawing.py llgrutil.py pyopengl_draw_instanced.py

OPENGL_SHADERS = fshader150.txt vshader150.txt

all: $(PYOBJS)

install: all
	-mkdir $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(PKG_DIR)

clean:
	rm -rf __pycache__
