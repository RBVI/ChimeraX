FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

DOCUTILS_VERSION = 0.10
DOCUTILS_PACKAGE = docutils-$(DOCUTILS_VERSION).tar.gz
DOCUTILS_SOURCE = $(tmpdir)/docutils-$(DOCUTILS_VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(DOCUTILS_SOURCE)
	cd $(DOCUTILS_SOURCE) && python setup.py $(SETUP_BUILD) install

$(DOCUTILS_SOURCE):
	tar zxf $(DOCUTILS_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(DOCUTILS_SOURCE)
