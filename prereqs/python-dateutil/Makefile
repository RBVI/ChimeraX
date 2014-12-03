FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.3
DISTRIBUTION = python-dateutil-$(VERSION).tar.gz
SOURCE = $(tmpdir)/python-dateutil-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	python3 setup.py install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
