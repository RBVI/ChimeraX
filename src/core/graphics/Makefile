TOP = ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/graphics
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/graphics

SUBDIRS	= _graphics

PYSRCS = __init__.py camera.py crossfade.py drawing.py opengl.py view.py

OPENGL_SHADERS = fragmentShader.txt vertexShader.txt

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all dir
	$(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(PKG_DIR)
	if [ -d "$(APP_PKG_DIR)" ]; then $(RSYNC) $(PYSRCS) $(OPENGL_SHADERS) $(APP_PKG_DIR); fi

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi
	if [ -d "$(app_prefix)" ]; then mkdir -p $(APP_PKG_DIR); fi

clean:
	rm -rf __pycache__

lint:
	$(PYLINT) $(PYSRCS)
