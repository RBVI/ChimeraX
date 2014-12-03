FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.2.5
DISTRIBUTION = flake8-$(VERSION).tar.gz
SOURCE = $(tmpdir)/flake8-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	python3 setup.py install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
