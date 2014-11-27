TOP = ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/graphics

SUBDIRS	= graphics_cpp

PYSRCS = __init__.py camera.py crossfade.py drawing.py opengl.py view.py

OPENGL_SHADERS = fragmentShader.txt vertexShader.txt

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR)
	$(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(PKG_DIR)

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__
