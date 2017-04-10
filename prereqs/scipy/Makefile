PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

VERSION = 0.19.0

ifdef WIN32
# Downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/
WHEEL = scipy-$(VERSION)-cp36-cp36m-win_amd64.whl
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
