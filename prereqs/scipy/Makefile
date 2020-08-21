PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

VERSION = 1.4.1

ifdef WIN32
# Downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/
WHEEL = scipy-$(VERSION)-cp$(PYVER_NODOT)-cp$(PYVER_NODOT)m-win_amd64.whl
endif

all:

ifdef WIN32

install:
	$(PIP_INSTALL) $(WHEEL)

app-install: install
	$(APP_PIP_INSTALL) $(WHEEL)

$(WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/scipy/$(WHEEL)

upload_new_version:
	$(RSYNC) $(WHEEL) $(PREREQS_UPLOAD)/scipy

else

install:
	$(PIP_INSTALL) scipy==$(VERSION)

app-install: install
	$(APP_PIP_INSTALL) -U scipy==$(VERSION)

endif

clean:
	rm -rf pip.log
