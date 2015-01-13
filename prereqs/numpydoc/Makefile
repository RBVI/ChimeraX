FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.5
DISTRIBUTION = numpydoc-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numpydoc-$(VERSION)

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
	$(RSYNC) $(PYSITEDIR)/numpydoc-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
