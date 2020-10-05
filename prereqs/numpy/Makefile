PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

VERSION = 1.18.5

ifdef WIN32
# Downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/
WHEEL = numpy-$(VERSION)+mkl-cp$(PYVER_NODOT)-cp$(PYVER_NODOT)m-win_amd64.whl
endif

all:

ifdef WIN32

install: $(WHEEL)
	$(PIP_INSTALL) $(WHEEL)

app-install: install
	$(APP_PIP_INSTALL) $(WHEEL)

$(WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/numpy/$(WHEEL)

upload_new_version:
	$(RSYNC) $(WHEEL) $(PREREQS_UPLOAD)/numpy

else

install:
	$(PIP_INSTALL) numpy==$(VERSION)

app-install: install
	$(APP_PIP_INSTALL) numpy==$(VERSION)

endif

clean:
	rm -rf pip.log
