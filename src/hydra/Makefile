TOP = ../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra
APPS_DIR = $(build_prefix)/Applications

PYSRCS = __init__.py align.py bins.py biomt.py c2leap.py colors.py commands.py \
	fetch.py fetch_emdb.py fetch_pdb.py gui.py history.py icosahedron.py \
	matrix.py molecule.py molmap.py opensave.py pdb.py place.py pyopengl_draw_instanced.py \
	qt.py read_apr.py readstl.py session.py shaders.py shortcuts.py spline.py \
	surface.py tube.py vector.py view.py FitMap SessionUtil VolumeData VolumeViewer
PYLIBS = _image3d/_image3d.so

DATA = docs icons vshader150.txt fshader150.txt

INSTALL_DIRS = $(PKG_DIR)
ifeq ($(OS),Darwin)
  FW_DIR = $(APPS_DIR)/Hydra.app/Contents/Frameworks
  INSTALL_DIRS += $(APPS_DIR) $(FW_DIR) $(FW_DIR)/Python.framework
endif

all: $(PYLIBS)

install: all $(INSTALL_DIRS)
	$(RSYNC) $(PYSRCS) $(PYLIBS) $(DATA) $(PKG_DIR)
ifeq ($(OS),Darwin)
	$(RSYNC) Hydra.app $(APPS_DIR)
endif

$(PKG_DIR):
	-mkdir $(PKG_DIR)
$(APPS_DIR):
	-mkdir $(APPS_DIR)
$(FW_DIR):
	-mkdir $(FW_DIR)
$(FW_DIR)/Python.framework:
	-ln -s $(frameworkdir)/Python.framework $(FW_DIR)

run:
	$(bindir)/python3 -c "import hydra; hydra.show_main_window()"

_image3d/_image3d.so:
	$(MAKE) -C _image3d

clean:
	rm -rf __pycache__
