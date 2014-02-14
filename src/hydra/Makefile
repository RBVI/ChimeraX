TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	= docs draw example_sessions file_io geometry map molecule surface ui _image3d

PKG_DIR = $(PYSITEDIR)/hydra
APP_NAME = Hydra.app

PYSRCS = __init__.py models.py scenes.py session.py

INSTALL_DIRS = $(PKG_DIR)
ifeq ($(OS),Darwin)
  APPS_DIR = $(build_prefix)/Applications
  FW_DIR = $(APPS_DIR)/$(APP_NAME)/Contents/Frameworks
  PY_SITE_INSTALL=$(FW_DIR)/Python.framework/Versions/3.3/lib/python3.3/site-packages
# QtPrintSupport linked by libqcocoa.dylib Qt plugin
  QT_FRAMEWORKS = QtCore QtGui QtOpenGL QtPrintSupport QtWidgets
  QT_LIBS = $(foreach fw,$(QT_FRAMEWORKS),$(fw).framework/Versions/5/$(fw))
  PY_LIB = $(frameworkdir)/Python.framework/Versions/3.3/Python
  QT_PLUGINS = plugins/platforms/libqcocoa.dylib \
		plugins/imageformats/libqjpeg.dylib \
		plugins/accessible/libqtaccessiblewidgets.dylib
endif

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)

ifeq ($(OS),Darwin)
install_mac_app: install mac_app_dirs
# Build application directory hierarchy
	$(RSYNC) $(APP_NAME) $(APPS_DIR)
	$(RSYNC) --exclude site-packages $(frameworkdir)/Python.framework $(FW_DIR)

# Copy only the Python packages needed for distribution.
	cd $(libdir)/python3.3/site-packages ; $(RSYNC) numexpr-2.2.2-py3.3-macosx-10.8-x86_64.egg/numexpr numpy tables PyQt5 sip.so $(PY_SITE_INSTALL)
	cd $(PYSITEDIR) ; $(RSYNC) hydra LeapMotion PyOpenGL-3.1.0a1-py3.3.egg/OpenGL $(PY_SITE_INSTALL)

# Fix Python executable reference to python shared library
	install_name_tool -change $(PY_LIB) @executable_path/../Python $(FW_DIR)/Python.framework/Versions/3.3/bin/python3.3
	-chmod u+w $(FW_DIR)/Python.framework/Versions/3.3/Python
	install_name_tool -change $(PY_LIB) @executable_path/../../../../Python $(FW_DIR)/Python.framework/Versions/3.3/Resources/Python.app/Contents/MacOS/Python

# Copy Qt frameworks into PyQt5 python module
	cd $(libdir) ; $(RSYNC) $(QT_FRAMEWORKS:%=%.framework) $(PY_SITE_INSTALL)/PyQt5/frameworks
# Fix Qt framework paths to other Qt frameworks
	for qtlib0 in $(QT_LIBS) ; do \
		for qtlib in $(QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/../../../$$qtlib $(PY_SITE_INSTALL)/PyQt5/frameworks/$$qtlib0 ; \
		done ; \
	done

# Fix PyQt library paths to Qt frameworks
	for pyqtlib in $(PY_SITE_INSTALL)/PyQt5/*.so ; do \
		for qtlib in $(QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/frameworks/$$qtlib $$pyqtlib ; \
		done ; \
	done

# Install needed Qt plugins
	cd $(build_prefix) ; $(RSYNC) -R $(QT_PLUGINS) $(PY_SITE_INSTALL)/PyQt5
# Fix plugin library paths to Qt frameworks.
	for plugin in $(QT_PLUGINS) ; do \
		for qtlib in $(QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/../../frameworks/$$qtlib $(PY_SITE_INSTALL)/PyQt5/$$plugin ; \
		done ; \
	done

# Make PyTables link to libhdf5.8.dylib
	$(RSYNC) $(libdir)/libhdf5.8.dylib $(PY_SITE_INSTALL)/tables
	for tableslib in $(PY_SITE_INSTALL)/tables/*.so ; do \
		install_name_tool -change $(libdir)/libhdf5.8.dylib @loader_path/libhdf5.8.dylib $$tableslib ; \
	done

# Missing lib-dynload in Python framework
	$(RSYNC) $(libdir)/python3.3/lib-dynload $(FW_DIR)/Python.framework/Versions/3.3/lib/python3.3

# Fix _image3d.so path to python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/_image3d.so

# Fix _spacenavigator.so path to Python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PY_SITE_INSTALL)/hydra/ui/spacenavigator/_spacenavigator.so

# Fix _oculus.so path to Python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PY_SITE_INSTALL)/hydra/ui/oculus/_oculus.so

# Install pdbio and fix paths to libraries
	$(RSYNC) $(libdir)/libmolecule.dylib $(libdir)/libtmpl.dylib $(libdir)/libcapsule.dylib $(PY_SITE_INSTALL)/hydra/lib
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(PY_SITE_INSTALL)/hydra/lib/libcapsule.dylib
	$(RSYNC) $(PYSITEDIR)/pdbio.so $(PYSITEDIR)/access.so $(PY_SITE_INSTALL)/hydra
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/access.so 
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/pdbio.so
	for pymod in access.so pdbio.so ; do \
		for lib in libmolecule.dylib libcapsule.dylib libtmpl.dylib ; do \
			install_name_tool -change @executable_path/../lib/$$lib @loader_path/lib/$$lib $(PY_SITE_INSTALL)/hydra/$$pymod ; \
		done ; \
	done

mac_app_dirs:
	for d in $(APPS_DIR) $(FW_DIR) $(PY_SITE_INSTALL)/hydra/lib $(PY_SITE_INSTALL)/PyQt5/frameworks ; do \
		if [ ! -d "$$d" ]; then mkdir -p "$$d"; fi ; \
	done

endif

run:
	$(bindir)/python3 -c "import hydra; hydra.Session().start()"

dmg:
	cd $(APPS_DIR) ; \
	hdiutil create $(basename $(APP_NAME))-`date "+%Y-%m-%d"`.dmg -srcfolder $(APP_NAME) -volname $(basename $(APP_NAME)) -fs HFS+ -format UDBZ


clean:
	rm -rf __pycache__
