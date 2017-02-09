PREREQ_MAKE = 1
TOP	= ../../..
include $(TOP)/mk/config.make

SIP_VERSION = 4.18.1
DISTRIBUTION = sip-$(SIP_VERSION).tar.gz
SOURCE = $(tmpdir)/sip-$(SIP_VERSION)

ifdef WIN32
MAKE = nmake
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py && $(MAKE) && $(MAKE) install

app-install:
	$(RSYNC) $(PYSITEDIR)/sip.$(PYMOD_EXT) $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
