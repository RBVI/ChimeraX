FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 4.10.1
DISTRIBUTION = PyQt-win-gpl-$(VERSION).zip
SOURCE = $(tmpdir)/PyQt-win-gpl-$(VERSION)

all:

install: $(SOURCE)
	-mkdir $(PYSITEDIR)/PyQt4
	-mkdir -p $(datadir)/sip/PyQt4
	cd $(SOURCE) && python3 configure-ng.py --confirm-license && $(MAKE) && $(MAKE) install

$(SOURCE): $(DISTRIBUTION)
	unzip -d $(tmpdir) -q $(DISTRIBUTION)

clean:
	rm -rf $(SOURCE)
