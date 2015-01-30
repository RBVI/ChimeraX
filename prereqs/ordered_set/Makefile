FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.3
DISTRIBUTION = ordered-set-$(VERSION).tar.gz
SOURCE = $(tmpdir)/ordered-set-$(VERSION)

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
	$(RSYNC) $(PYSITEDIR)/ordered_set-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
