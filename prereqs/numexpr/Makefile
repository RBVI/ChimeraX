PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ifdef WIN32
VERSION = 2.6.2
else
VERSION = 2.6.2
# VERSION = 2.4.6
endif
DISTRIBUTION = numexpr-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numexpr-$(VERSION)

ifdef WIN32
include ../pips/Makefile.pip
WHEEL = numexpr-$(VERSION)-cp36-cp36m-win_amd64.whl
endif

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install: $(SOURCE)
ifdef WIN32
	$(PIP) install $(WHEEL)
else
	# cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	$(PIP) install numexpr==$(VERSION)
endif

app-install: install
ifdef WIN32
	$(APP_PIP) install $(WHEEL)
else
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/numexpr-$(VERSION)-py$(PYTHON_VERSION)-*.egg $(APP_PYSITEDIR)
	# cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install
	$(APP_PIP) install numexpr==$(VERSION)
endif

$(SOURCE):
ifndef WIN32
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
endif

clean:
	rm -rf $(SOURCE) pip.log
