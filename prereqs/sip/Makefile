FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 4.14.7-snapshot-74e1df1d9940
DISTRIBUTION = sip-$(VERSION).tar.gz
SOURCE = $(tmpdir)/sip-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) && python3 configure.py && $(MAKE) && $(MAKE) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
