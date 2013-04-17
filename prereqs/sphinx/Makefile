FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

SPHINX_VERSION = 1.2b1
SPHINX_PACKAGE = Sphinx-$(SPHINX_VERSION).tar.gz
SPHINX_SOURCE = $(tmpdir)/Sphinx-$(SPHINX_VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SPHINX_SOURCE)
	cd $(SPHINX_SOURCE) && python setup.py $(SETUP_BUILD) install

$(SPHINX_SOURCE):
	tar zxf $(SPHINX_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(SPHINX_SOURCE)
