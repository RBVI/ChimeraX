TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS	= arrays commands devices docs example_sessions file_io geometry graphics mac_os_cpp \
	  map map_cpp molecule molecule_cpp movie surface ui

PKG_DIR = $(PYSITEDIR)/hydra
APP_NAME = Hydra.app

PYSRCS = __init__.py models.py scenes.py session.py version.py

INSTALL_DIRS = $(PKG_DIR)
ifeq ($(OS),Darwin)
  APPS_DIR = $(build_prefix)/Applications
  APP = $(APPS_DIR)/$(APP_NAME)
  APP_PY_EXE = $(APP)/Contents/MacOS/python
  APP_FW_DIR = $(APP)/Contents/Frameworks
  APP_BIN_DIR = $(APP)/Contents/Resources/bin
  APP_PY_DIR = $(APP_FW_DIR)/Python.framework/Versions/$(PYTHON_VERSION)
  APP_PY_SITE = $(APP_PY_DIR)/lib/python$(PYTHON_VERSION)/site-packages
  APP_PYQT = $(APP_PY_SITE)/PyQt5
  APP_HYDRA = $(APP_PY_SITE)/hydra

# QtPrintSupport linked by libqcocoa.dylib Qt plugin
  QT_FRAMEWORKS = QtCore QtGui QtOpenGL QtPrintSupport QtWidgets
  ALL_QT_FRAMEWORKS = QtCLucene QtConcurrent QtCore QtDeclarative QtDesigner QtDesignerComponents QtGui QtHelp QtMultimedia QtMultimediaQuick_p QtMultimediaWidgets QtNetwork QtOpenGL QtPrintSupport QtQml QtQuick QtQuickParticles QtQuickTest QtScript QtScriptTools QtSensors QtSerialPort QtSql QtSvg QtTest QtV8 QtWebKit QtWebKitWidgets QtWidgets QtXml QtXmlPatterns
  QT_LIBS = $(foreach fw,$(QT_FRAMEWORKS),$(fw).framework/Versions/5/$(fw))
  ALL_QT_LIBS = $(foreach fw,$(ALL_QT_FRAMEWORKS),$(fw).framework/Versions/5/$(fw))
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
  SIX_VERSION = 1.6.1
  SIX_EGG = six-$(SIX_VERSION)-py$(PYTHON_VERSION).egg
endif

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: $(PYOBJS)

install: all $(PKG_DIR) install_app_prereqs
	$(RSYNC) $(PYSRCS) $(PKG_DIR)
ifeq ($(OS),Darwin)
	$(RSYNC) $(PYSITEDIR)/hydra $(APP_PY_SITE)
endif

ifeq ($(OS),Darwin)

install_app_prereqs: $(APP)

$(APP) update_prereqs:

# Make directories
	-mkdir -p $(APPS_DIR) $(APP_FW_DIR)

# Build application directory hierarchy
	$(RSYNC) $(APP_NAME) $(APPS_DIR)

# Copy Python
	$(RSYNC) $(bindir)/python$(PYTHON_VERSION)noapp $(APP_PY_EXE)
	install_name_tool -change $(PY_LIB) @executable_path/../Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/Python $(APP_PY_EXE)
# Python 3.4 framework has symbolic link to the non-framework lib/python3.4.
# To copy that we use rsync option -copy-unsafe-links to copy links outside the tree.
	$(RSYNC) --copy-unsafe-links --exclude site-packages $(frameworkdir)/Python.framework $(APP_FW_DIR)
	install_name_tool -change $(PY_LIB) @executable_path/../../../../Python $(APP_PY_DIR)/Resources/Python.app/Contents/MacOS/Python

# Copy only the Python packages needed for distribution.
	cd $(libdir)/python$(PYTHON_VERSION)/site-packages ; \
	  $(RSYNC) $(DATEUTIL_EGG)/dateutil LeapMotion $(NUMEXPR_EGG)/numexpr numpy PIL \
	           $(PYCOLLADA_EGG)/collada $(PYOPENGL_EGG)/OpenGL PyQt5 sip.so $(SIX_EGG)/six.py \
		   tables webcolors.py $(APP_PY_SITE)

# Copy Qt frameworks into PyQt5 python module
	-mkdir $(APP_PYQT)/frameworks
	cd $(libdir) ; $(RSYNC) $(QT_FRAMEWORKS:%=%.framework) $(APP_PYQT)/frameworks
# Change id name of Qt libs to not point to build location.
	for qtlib in $(QT_LIBS) ; do \
		install_name_tool -id $$(basename $$qtlib) $(APP_PYQT)/frameworks/$$qtlib ; \
	done
# Fix Qt framework paths to other Qt frameworks
	for qtlib0 in $(QT_LIBS) ; do \
		for qtlib in $(ALL_QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/../../../$$qtlib $(APP_PYQT)/frameworks/$$qtlib0 ; \
		done ; \
	done

# Fix PyQt library paths to Qt frameworks
	for pyqtlib in $(APP_PYQT)/*.so ; do \
		for qtlib in $(ALL_QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/frameworks/$$qtlib $$pyqtlib ; \
		done ; \
	done

# Install needed Qt plugins
	cd $(build_prefix) ; $(RSYNC) -R $(QT_PLUGINS) $(APP_PYQT)
# Fix plugin library paths to Qt frameworks.
	for plugin in $(QT_PLUGINS) ; do \
		for qtlib in $(QT_LIBS) ; do \
			install_name_tool -change $(libdir)/$$qtlib @loader_path/../../frameworks/$$qtlib $(APP_PYQT)/$$plugin ; \
		done ; \
	done

# Make PyTables link to libhdf.dylib
	$(RSYNC) $(libdir)/libhdf$(HDF_VERSION).dylib $(APP_PY_SITE)/tables
	for tableslib in $(APP_PY_SITE)/tables/*.so ; do \
		install_name_tool -change $(libdir)/libhdf$(HDF_VERSION).dylib @loader_path/libhdf$(HDF_VERSION).dylib $$tableslib ; \
	done

# Install pdbio and fix paths to libraries
	-mkdir -p $(APP_HYDRA)/lib
	$(RSYNC) $(libdir)/libatomstruct.dylib $(libdir)/libtmpl.dylib $(libdir)/libblob.dylib $(APP_HYDRA)/lib
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(APP_HYDRA)/lib/libblob.dylib
	install_name_tool -change @executable_path/../lib/libatomstruct.dylib $(APP_HYDRA)/lib/libatomstruct.dylib $(APP_HYDRA)/lib/libblob.dylib
	install_name_tool -change @executable_path/../lib/libtmpl.dylib $(APP_HYDRA)/lib/libtmpl.dylib $(APP_HYDRA)/lib/libatomstruct.dylib
	$(RSYNC) $(PYSITEDIR)/pdbio.so $(PYSITEDIR)/structaccess.so $(APP_HYDRA)
	# Work around pdbio.parse_mmCIF_file() importing structaccess globally.
	echo "from hydra.structaccess import *" >  $(APP_PY_SITE)/structaccess.py
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(APP_HYDRA)/structaccess.so 
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(APP_HYDRA)/pdbio.so
	for pymod in structaccess.so pdbio.so ; do \
		for lib in libatomstruct.dylib libblob.dylib libtmpl.dylib libconnectivity.dylib ; do \
			install_name_tool -change @executable_path/../lib/$$lib @loader_path/lib/$$lib $(APP_HYDRA)/$$pymod ; \
		done ; \
	done

# Copy ffmpeg
	-mkdir -p $(APP_BIN_DIR)
	$(RSYNC) $(bindir)/ffmpeg $(APP_BIN_DIR)

dmg:
	cd $(APPS_DIR) ; \
	hdiutil create $(basename $(APP_NAME))-`date "+%Y-%m-%d"`.dmg -srcfolder $(APP_NAME) -volname $(basename $(APP_NAME)) -fs HFS+ -format UDBZ

else

install_app_prereqs: 

endif

$(PKG_DIR):
	if [ ! -d "$(PKG_DIR)" ]; then mkdir $(PKG_DIR); fi

clean:
	rm -rf __pycache__

map_cpp.all: arrays.install
molecule_cpp.all: arrays.install
