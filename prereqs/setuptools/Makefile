FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

SETUPTOOLS_VERSION = 0.6c11
SETUPTOOLS_PACKAGE = setuptools-$(SETUPTOOLS_VERSION).tar.gz
SETUPTOOLS_SOURCE = $(tmpdir)/setuptools-$(SETUPTOOLS_VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SETUPTOOLS_SOURCE)
	cd $(SETUPTOOLS_SOURCE) && python setup.py $(SETUP_BUILD) install

$(SETUPTOOLS_SOURCE):
	tar zxf $(SETUPTOOLS_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(SETUPTOOLS_SOURCE)
