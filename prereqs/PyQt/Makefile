PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Methods are "wheel.*" and "fromsrc"
METHOD=wheel.qt6

ifeq ($(OSARCH),DarwinArm64)
METHOD=pypi_qt6
endif

# PYQT_LICENSE can be "commercial" for commercial version
PYQT_LICENSE = gpl

USE_QT6 = true

BASE_VERSION = 6.5
QT_MINOR_VERSION = 1
PYQT_MINOR_VERSION = 1
PYQTWE_MINOR_VERSION = 0
METHOD=wheel.qt6

MAKE_ARGS = -f Makefile.$(METHOD) BASE_VERSION=$(BASE_VERSION) PYQT_MINOR_VERSION=$(PYQT_MINOR_VERSION) QT_MINOR_VERSION=$(QT_MINOR_VERSION) PYQTWE_MINOR_VERSION=$(PYQTWE_MINOR_VERSION) PYQT_LICENSE=$(PYQT_LICENSE)

all:

install:
	$(MAKE) $(MAKE_ARGS) install

app-install:
	$(MAKE) $(MAKE_ARGS) app-install

clean:
	$(MAKE) $(MAKE_ARGS) clean

distclean: clean
	$(MAKE) $(MAKE_ARGS) distclean
