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
# QtPrintSupport linked by libqcocoa.dylib Qt plugin
  QT_FRAMEWORKS = QtCore QtGui QtOpenGL QtPrintSupport QtWidgets
  ALL_QT_FRAMEWORKS = QtCLucene QtConcurrent QtCore QtDeclarative QtDesigner QtDesignerComponents QtGui QtHelp QtMultimedia QtMultimediaQuick_p QtMultimediaWidgets QtNetwork QtOpenGL QtPrintSupport QtQml QtQuick QtQuickParticles QtQuickTest QtScript QtScriptTools QtSensors QtSerialPort QtSql QtSvg QtTest QtV8 QtWebKit QtWebKitWidgets QtWidgets QtXml QtXmlPatterns
  QT_LIBS = $(foreach fw,$(QT_FRAMEWORKS),$(fw).framework/Versions/5/$(fw))
  ALL_QT_LIBS = $(foreach fw,$(ALL_QT_FRAMEWORKS),$(fw).framework/Versions/5/$(fw))
  PY_INSTALL=$(FW_DIR)/Python.framework/Versions/$(PYTHON_VERSION)
  PY_SITE_INSTALL=$(PY_INSTALL)/lib/python$(PYTHON_VERSION)/site-packages
  PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
  QT_PLUGINS = plugins/platforms/libqcocoa.dylib \
		plugins/imageformats/libqjpeg.dylib \
		plugins/accessible/libqtaccessiblewidgets.dylib
  PYOPENGL_VERSION = 3.1.0a1
  PYOPENGL_EGG = PyOpenGL-$(PYOPENGL_VERSION)-py$(PYTHON_VERSION).egg
  NUMEXPR_VERSION = 2.3.1
  NUMEXPR_EGG = numexpr-$(NUMEXPR_VERSION)-py$(PYTHON_VERSION)-macosx-$(MACOSX_DEPLOYMENT_TARGET)-x86_64.egg
  PYCOLLADA_VERSION = 0.4
  PYCOLLADA_EGG = pycollada-$(PYCOLLADA_VERSION)-py$(PYTHON_VERSION).egg
  DATEUTIL_VERSION = 2.2
  DATEUTIL_EGG = python_dateutil-$(DATEUTIL_VERSION)-py$(PYTHON_VERSION).egg
  HDF_VERSION = 5.8
endif

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all pkg_dir
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
ifeq ($(OS),Darwin)
	$(MAKE) install_mac_app
endif

ifeq ($(OS),Darwin)

install_mac_app: mac_app_dirs
# Build application directory hierarchy
	$(RSYNC) $(APP_NAME) $(APPS_DIR)

# Copy Python
	$(RSYNC) $(bindir)/python$(PYTHON_VERSION)noapp $(APPS_DIR)/$(APP_NAME)/Contents/MacOS/python
	install_name_tool -change $(PY_LIB) @executable_path/../Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/Python $(APPS_DIR)/$(APP_NAME)/Contents/MacOS/python
# Python 3.4 framework has symbolic link to the non-framework lib/python3.4.
# To copy that we use rsync option -copy-unsafe-links to copy links outside the tree.
	$(RSYNC) --copy-unsafe-links --exclude site-packages $(frameworkdir)/Python.framework $(FW_DIR)
	-chmod u+w $(PY_INSTALL)/Python
	install_name_tool -change $(PY_LIB) @executable_path/../../../../Python $(PY_INSTALL)/Resources/Python.app/Contents/MacOS/Python

# Copy only the Python packages needed for distribution.
	cd $(libdir)/python$(PYTHON_VERSION)/site-packages ; \
	  $(RSYNC) $(NUMEXPR_EGG)/numexpr numpy tables PyQt5 sip.so $(PY_SITE_INSTALL)
	cd $(PYSITEDIR) ; \
	  $(RSYNC) hydra LeapMotion $(PYCOLLADA_EGG)/collada $(PYOPENGL_EGG)/OpenGL \
	           $(DATEUTIL_EGG)/dateutil $(PY_SITE_INSTALL)

# Copy Qt frameworks into PyQt5 python module
	cd $(libdir) ; $(RSYNC) $(QT_FRAMEWORKS:%=%.framework) $(PY_SITE_INSTALL)/PyQt5/frameworks
# Change id name of Qt libs to not point to build location.
	for qtlib in $(QT_LIBS) ; do \
		install_name_tool -id $$(basename $$qtlib) $(PY_SITE_INSTALL)/PyQt5/frameworks/$$qtlib ; \
	done
# Fix Qt framework paths to other Qt frameworks
	for qtlib0 in $(QT_LIBS) ; do \
		for qtlib in $(ALL_QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/../../../$$qtlib $(PY_SITE_INSTALL)/PyQt5/frameworks/$$qtlib0 ; \
		done ; \
	done

# Fix PyQt library paths to Qt frameworks
	for pyqtlib in $(PY_SITE_INSTALL)/PyQt5/*.so ; do \
		for qtlib in $(ALL_QT_LIBS) ; do \
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

# Make PyTables link to libhdf.dylib
	$(RSYNC) $(libdir)/libhdf$(HDF_VERSION).dylib $(PY_SITE_INSTALL)/tables
	for tableslib in $(PY_SITE_INSTALL)/tables/*.so ; do \
		install_name_tool -change $(libdir)/libhdf$(HDF_VERSION).dylib @loader_path/libhdf$(HDF_VERSION).dylib $$tableslib ; \
	done

# Missing lib-dynload in Python framework
	$(RSYNC) $(libdir)/python$(PYTHON_VERSION)/lib-dynload $(PY_INSTALL)/lib/python$(PYTHON_VERSION)

# Fix _image3d.so path to python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/_image3d.so

# Fix _spacenavigator.so path to Python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PY_SITE_INSTALL)/hydra/ui/spacenavigator/_spacenavigator.so

# Fix _oculus.so path to Python library
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../../Python $(PY_SITE_INSTALL)/hydra/ui/oculus/_oculus.so

# Install pdbio and fix paths to libraries
	$(RSYNC) $(libdir)/libatomstruct.dylib $(libdir)/libtmpl.dylib $(libdir)/libblob.dylib $(PY_SITE_INSTALL)/hydra/lib
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(PY_SITE_INSTALL)/hydra/lib/libblob.dylib
	$(RSYNC) $(PYSITEDIR)/pdbio.so $(PYSITEDIR)/structaccess.so $(PY_SITE_INSTALL)/hydra
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/structaccess.so 
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PY_SITE_INSTALL)/hydra/pdbio.so
	for pymod in structaccess.so pdbio.so ; do \
		for lib in libatomstruct.dylib libblob.dylib libtmpl.dylib ; do \
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
