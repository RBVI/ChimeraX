FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.4.4
DISTRIBUTION = numexpr-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numexpr-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install: install
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/numexpr-$(VERSION)-py$(PYTHON_VERSION)-*.egg $(APP_PYSITEDIR)
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
