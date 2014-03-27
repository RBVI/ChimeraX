FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2014-3-20
DISTRIBUTION = pycollada-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pycollada-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	python3 setup.py install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
