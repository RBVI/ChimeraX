TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	= docs draw file_io FitMap geometry molecule ui VolumeData VolumeViewer _image3d

PKG_DIR = $(PYSITEDIR)/hydra
APPS_DIR = $(build_prefix)/Applications

PYSRCS = __init__.py molmap.py scenes.py surface.py

INSTALL_DIRS = $(PKG_DIR)
ifeq ($(OS),Darwin)
  FW_DIR = $(APPS_DIR)/Hydra.app/Contents/Frameworks
  INSTALL_DIRS += $(APPS_DIR) $(FW_DIR) $(FW_DIR)/Python.framework
endif

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(INSTALL_DIRS) install_mac_app
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

install_mac_app:
ifeq ($(OS),Darwin)
	$(RSYNC) Hydra.app $(APPS_DIR)
endif

$(PKG_DIR):
	-mkdir $(PKG_DIR)
$(APPS_DIR):
	-mkdir $(APPS_DIR)
$(FW_DIR):
	-mkdir -p $(FW_DIR)
$(FW_DIR)/Python.framework:
	-ln -s $(frameworkdir)/Python.framework $(FW_DIR)

run:
	$(bindir)/python3 -c "import hydra; hydra.show_main_window()"

clean:
	rm -rf __pycache__
