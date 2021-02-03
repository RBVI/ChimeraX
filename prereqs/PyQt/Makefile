PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# alternate method is "fromsrc"
METHOD=wheel

# PYQT_LICENSE can be "commercial" for commercial version
PYQT_LICENSE = gpl

VERSION = 5.15
PYQT_MINOR_VERSION = 2
QT_MINOR_VERSION = 2

MAKE_ARGS = -f Makefile.$(METHOD) VERSION=$(VERSION) PYQT_MINOR_VERSION=$(PYQT_MINOR_VERSION) QT_MINOR_VERSION=$(QT_MINOR_VERSION) PYQT_LICENSE=$(PYQT_LICENSE)

all:

install:
	$(MAKE) $(MAKE_ARGS) install

app-install:
	$(MAKE) $(MAKE_ARGS) app-install

clean:
	$(MAKE) $(MAKE_ARGS) clean

distclean: clean
	$(MAKE) $(MAKE_ARGS) distclean
