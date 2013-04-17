TOP	= ../..
include $(TOP)/mk/config.make

FREEFONT_VERSION = 20120719
FREEFONT_PACKAGE = freefont-ttf-$(FREEFONT_VERSION).tar.gz
FREEFONT_SOURCE = $(tmpdir)/freefont-$(FREEFONT_VERSION)

all:

install: $(FREEFONT_SOURCE)
	-mkdir $(datadir)/fonts
	-mkdir $(datadir)/fonts/freefont
	$(RSYNC) $(FREEFONT_SOURCE)/* $(datadir)/fonts/freefont

$(FREEFONT_SOURCE):
	tar zxf $(FREEFONT_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(FREEFONT_SOURCE)
