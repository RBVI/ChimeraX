FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

JINJA2_VERSION = 2.6
JINJA2_PACKAGE = Jinja2-$(JINJA2_VERSION).tar.gz
JINJA2_SOURCE = $(tmpdir)/Jinja2-$(JINJA2_VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(JINJA2_SOURCE)
	cd $(JINJA2_SOURCE) && python setup.py $(SETUP_BUILD) install

$(JINJA2_SOURCE):
	tar zxf $(JINJA2_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(JINJA2_SOURCE)
