FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 7.0
DISTRIBUTION = setuptools-$(VERSION).tar.gz
SOURCE = $(tmpdir)/setuptools-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install:
	cd $(PYSITEDIR); $(RSYNC) \
		setuptools-$(VERSION)-py$(PYTHON_VERSION).egg \
		easy-install.pth easy_install.py pkg_resources.py _markerlib \
		$(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
