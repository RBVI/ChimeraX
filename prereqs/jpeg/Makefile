FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 9a
DISTRIBUTION = jpegsrc.v$(VERSION).tar.gz
SOURCE = $(tmpdir)/jpeg-$(VERSION)

install: $(SOURCE)
	cd $(SOURCE) ; ./configure --prefix=$(build_prefix) --disable-static && make install-libLTLIBRARIES

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
