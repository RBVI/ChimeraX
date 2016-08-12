PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ifdef WIN32
FROM_WHEEL = 1
endif
PYQT_LICENSE = commercial
REMOTE_DIR = plato.cgl.ucsf.edu:/usr/local/src/PyQt5

VERSION = 5.7
DISTRIBUTION = PyQt5_$(PYQT_LICENSE)-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt5_$(PYQT_LICENSE)-$(VERSION)

ifeq ($(OS),Darwin)
SIP_BIN = $(PYTHON_FRAMEWORK)/bin
QT_BIN = $(bindir)
else ifdef WIN32
SIP_BIN = $(shell cygpath -m '$(bindir)')
QT_BIN = c:/Qt/Qt5.6.0/5.6/msvc2015_64/bin
else
SIP_BIN = $(bindir)
QT_BIN = $(bindir)
endif
CONFIG_ARGS = --confirm-license --verbose --sip $(SIP_BIN)/sip$(PROG_EXT) --qmake $(QT_BIN)/qmake$(PROG_EXT)

PATCHES = $(wildcard PyQt-*.patch)

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(frameworkdir)", "$(bindir)"))')

all:

ifdef FROM_WHEEL
install:
	$(MAKE) -f Makefile.wheel VERSION=$(VERSION) install
else
install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install
endif

app-install:
ifdef FROM_WHEEL
	$(MAKE) -f Makefile.wheel VERSION=$(VERSION) app-install
else
	$(RSYNC) $(PYSITEDIR)/PyQt5 $(APP_PYSITEDIR)
endif

distribution: $(DISTRIBUTION)

$(DISTRIBUTION):
ifeq ($(PYQT_LICENSE),commercial)
	rsync -a $(REMOTE_DIR)/*commercial* .
endif

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifeq ($(PYQT_LICENSE),commercial)
	cp pyqt-commercial.sip $(SOURCE)/sip/
endif
ifneq (,$(PATCHES))
	for f in $(PATCHES) ; do \
		( cd $(SOURCE) && patch -p0 ) < $$f ; \
	done
endif

clean:
ifdef FROM_WHEEL
	$(MAKE) -f Makefile.wheel VERSION=$(VERSION) clean
else
	rm -rf $(SOURCE)
endif

distclean: clean
ifdef FROM_WHEEL
	$(MAKE) -f Makefile.wheel VERSION=$(VERSION) distclean
else
	rm -f *commercial*
endif
