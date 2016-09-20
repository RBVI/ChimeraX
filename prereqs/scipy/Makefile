PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

VERSION = 0.18.0

ifdef WIN32
WHEEL = scipy-$(VERSION)-cp35-cp35m-win_amd64.whl
endif

all:

install:
ifdef WIN32
	$(PIP) install -U $(WHEEL)
else
	$(PIP) install -U scipy==$(VERSION)
endif

app-install: install
ifdef WIN32
	$(APP_PIP) install $(WHEEL)
else
	$(APP_PIP) install -U scipy==$(VERSION)
endif

clean:
	rm -rf pip.log
