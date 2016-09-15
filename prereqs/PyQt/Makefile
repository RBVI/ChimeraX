PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# alternate method is "fromsrc"
METHOD=wheel

# PYQT_LICENSE can be "gpl" for gpl version
PYQT_LICENSE = commercial

VERSION = 5.7
MINOR_VERSION = 0

MAKE_ARGS = -f Makefile.$(METHOD) VERSION=$(VERSION) MINOR_VERSION=$(MINOR_VERSION) PYQT_LICENSE=$(PYQT_LICENSE)

all:

install:
	$(MAKE) $(MAKE_ARGS) install

app-install:
	$(MAKE) $(MAKE_ARGS) app-install

clean:
	$(MAKE) $(MAKE_ARGS) clean

distclean: clean
	$(MAKE) $(MAKE_ARGS) distclean
