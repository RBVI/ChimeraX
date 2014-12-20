TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 20120719
DISTRIBUTION = freefont-ttf-$(VERSION).tar.gz
SOURCE = $(tmpdir)/freefont-$(VERSION)

all:

install: $(SOURCE)
	-mkdir -p $(datadir)/fonts/freefont
	$(RSYNC) $(SOURCE)/* $(datadir)/fonts/freefont

app-install:
	-mkdir -p $(app_datadir)/fonts/freefont
	$(RSYNC) $(datadir)/fonts/freefont $(app_datadir)/fonts

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
