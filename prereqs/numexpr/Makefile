FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.4
DISTRIBUTION = numexpr-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numexpr-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	python3 setup.py install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
