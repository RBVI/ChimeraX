FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 6b
DISTRIBUTION = jpeg-$(VERSION).tar.gz
SOURCE = $(tmpdir)/jpeg-$(VERSION)

install: $(SOURCE)
	cd $(SOURCE) ; ./configure --prefix=$(build_prefix) && make install-lib

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
