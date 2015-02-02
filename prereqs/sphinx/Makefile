FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.3b2
DISTRIBUTION = Sphinx-$(VERSION).tar.gz
SOURCE = $(tmpdir)/Sphinx-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
