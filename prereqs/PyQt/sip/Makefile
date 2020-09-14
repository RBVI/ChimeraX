PREREQ_MAKE = 1
TOP	= ../../..
include $(TOP)/mk/config.make

SIP_VERSION = 4.19.17
DISTRIBUTION = sip-$(SIP_VERSION).tar.gz
SOURCE = $(tmpdir)/sip-$(SIP_VERSION)

ifdef WIN32
MAKE = nmake
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py --sip-module PyQt5.sip && $(MAKE) && $(MAKE) install

app-install:
	$(RSYNC) $(PYSITEDIR)/sip.$(PYMOD_EXT) $(APP_PYSITEDIR)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/sip/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/sip

clean:
	rm -rf $(SOURCE)
