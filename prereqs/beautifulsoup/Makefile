FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 4.3.2
DISTRIBUTION = beautifulsoup4-$(VERSION).tar.gz
SOURCE = $(tmpdir)/beautifulsoup4-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) setup.py $(SETUP_BUILD) install

app-install:
	$(RSYNC) $(PYSITEDIR)/beautifulsoup4-$(VERSION)-py$(PYTHON_VERSION).egg-info $(PYSITEDIR)/bs4 $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
