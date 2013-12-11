FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.1.1
DISTRIBUTION = PyQt-gpl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt-gpl-$(VERSION)

CONFIG_ARGS = --confirm-license --verbose

all:

install: $(SOURCE)
	cd $(SOURCE) && python3 configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
