FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.1.0-rc1
DISTRIBUTION = xdg-utils-$(VERSION).tar.gz
SOURCE = $(tmpdir)/xdg-utils-$(VERSION)

all:

install: $(SOURCE)
	-mkdir $(datadir)/xdg-utils
	$(RSYNC) $(SOURCE)/scripts/xdg-* $(datadir)/xdg-utils

app-install:;
	-mkdir $(app_datadir)/xdg-utils
	$(RSYNC) $(SOURCE)/scripts/xdg-* $(app_datadir)/xdg-utils

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
