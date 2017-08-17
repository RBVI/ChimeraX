PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

VERSION = 1.13.1

ifdef WIN32
# Downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/
WHEEL = numpy-$(VERSION)+mkl-cp36-cp36m-win_amd64.whl
endif

all:

install:
ifdef WIN32
	$(PIP_INSTALL) $(WHEEL)
else
	$(PIP_INSTALL) numpy==$(VERSION)
endif

app-install: install
ifdef WIN32
	$(APP_PIP_INSTALL) $(WHEEL)
else
	$(APP_PIP_INSTALL) numpy==$(VERSION)
endif

clean:
	rm -rf pip.log
