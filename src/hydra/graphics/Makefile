TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/graphics

PYSRCS = __init__.py drawing.py llgrutil.py opengl.py pyopengl_draw_instanced.py

OPENGL_SHADERS = fshader150.txt vshader150.txt

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
