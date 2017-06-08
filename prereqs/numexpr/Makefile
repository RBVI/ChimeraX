PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

ifdef WIN32
VERSION = 2.6.2
else
VERSION = 2.6.2
endif
DISTRIBUTION = numexpr-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numexpr-$(VERSION)

ifdef WIN32
WHEEL = numexpr-$(VERSION)-cp36-cp36m-win_amd64.whl
endif

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install:
ifdef WIN32
	$(PIP_INSTALL) $(WHEEL)
else
	$(PIP_INSTALL) numexpr==$(VERSION)
endif

app-install: install
ifdef WIN32
	$(APP_PIP_INSTALL) $(WHEEL)
else
	$(APP_PIP_INSTALL) numexpr==$(VERSION)
endif

clean:
	rm -rf $(SOURCE) pip.log
