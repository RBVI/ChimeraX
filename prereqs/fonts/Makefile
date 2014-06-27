TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 20120719
DISTRIBUTION = freefont-ttf-$(VERSION).tar.gz
SOURCE = $(tmpdir)/freefont-$(VERSION)

all:

install: $(SOURCE)
	-mkdir $(datadir)/fonts
	-mkdir $(datadir)/fonts/freefont
	$(RSYNC) $(SOURCE)/* $(datadir)/fonts/freefont

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
