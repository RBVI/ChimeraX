TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/graphics

PYSRCS = __init__.py camera.py crossfade.py drawing.py opengl.py view.py

OPENGL_SHADERS = fragmentShader.txt vertexShader.txt

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
