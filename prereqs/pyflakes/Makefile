FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.8.1
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
	cd $(SOURCE) && $(PYTHON_EXE) setup.py $(SETUP_BUILD) install
ifdef USE_MAC_FRAMEWORKS
	cp -p $(PYTHON_FRAMEWORK)/bin/pyflakes $(bindir)
endif

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
