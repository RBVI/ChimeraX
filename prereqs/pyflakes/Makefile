FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.7.3
DISTRIBUTION = pyflakes-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pyflakes-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && python3 setup.py $(SETUP_BUILD) install
ifdef USE_MAC_FRAMEWORKS
	cp -p $(PYTHON_FRAMEWORK)/bin/pyflakes $(bindir)
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
