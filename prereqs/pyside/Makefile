# http://qt-project.org/wiki/Building_PySide
FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

SHIBOKEN_VERSION = 1.1.2
SHIBOKEN_PACKAGE = shiboken-$(SHIBOKEN_VERSION).tar.bz2
SHIBOKEN_SOURCE = $(tmpdir)/shiboken-$(SHIBOKEN_VERSION)

PYSIDE_VERSION = qt4.8+1.1.2
PYSIDE_PACKAGE = pyside-$(PYSIDE_VERSION).tar.bz2
PYSIDE_SOURCE = $(tmpdir)/pyside-$(PYSIDE_VERSION)

CMAKE_ENV = env CC='gcc -I$(includedir)' CXX='g++ -I$(includedir)'

CMAKE_OPTS = -DBUILD_TESTS=OFF \
	-DCMAKE_INSTALL_PREFIX=$(build_prefix) \
	-DCMAKE_PREFIX_PATH=$(build_prefix) \
	-DQT_INCLUDE_DIRS_NO_SYSTEM=1
ifdef DEBUG
CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Debug
else
CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Release
endif
ifdef USE_MAC_FRAMEWORKS
CMAKE_ENV = env CC='gcc -I$(includedir) -F$(frameworkdir)' \
	CXX='g++ -I$(includedir) -F$(frameworkdir)'
CMAKE_OPTS += -DQT_INCLUDE_DIR="$(includedir)"
CMAKE_OPTS += -DQT_USE_FRAMEWORKS=1
CMAKE_OPTS += -DCMAKE_INCLUDE_PATH="$(includedir)"
CMAKE_OPTS += -DCMAKE_FRAMEWORK_PATH="$(frameworkdir)"
CMAKE_OPTS += -DCMAKE_SKIP_INSTALL_RPATH=1
#CMAKE_OPTS += --Wdev --check-system-vars --trace -DCMAKE_VERBOSE_MAKEFILE=0
endif
#CMAKE_OPTS += -DCMAKE_OSX_DEPLOYMENT_TARGET=""

SHIBOKEN_OPTS = \
	-DPYTHON_INCLUDE_DIR=$(includedir)/python$(PYTHON_VERSION) \
	-DLIBXML2_LIBRARIES=$(libdir)/libxml2.$(SHLIB_EXT) \
	-DLIBXSLT_LIBRARIES=$(libdir)/libxslt.$(SHLIB_EXT)

all:

install: shiboken.install pyside.install

.NOTPARALLEL: shiboken.install pyside.install
	
shiboken.install: $(SHIBOKEN_SOURCE)
	cd $(SHIBOKEN_SOURCE)/build && $(MAKE) install

$(SHIBOKEN_SOURCE): $(SHIBOKEN_PACKAGE)
	tar jxf $(SHIBOKEN_PACKAGE) -C $(tmpdir)
	mkdir $(SHIBOKEN_SOURCE)/build
	cd $(SHIBOKEN_SOURCE)/build && $(CMAKE_ENV) cmake $(CMAKE_OPTS) $(SHIBOKEN_OPTS) ..
	
pyside.install: shiboken.install $(PYSIDE_SOURCE)
	cd $(PYSIDE_SOURCE)/build && $(MAKE) install

$(PYSIDE_SOURCE): $(PYSIDE_PACKAGE)
	tar jxf $(PYSIDE_PACKAGE) -C $(tmpdir)
	mkdir $(PYSIDE_SOURCE)/build
	cd $(PYSIDE_SOURCE)/build && $(CMAKE_ENV) cmake $(CMAKE_OPTS) ..

clean:
	rm -rf $(SHIBOKEN_SOURCE) $(PYSIDE_SOURCE)
