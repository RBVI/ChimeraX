FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 9a
LIBJPEG_VER = 9
DISTRIBUTION = jpegsrc.v$(VERSION).tar.gz
SOURCE = $(tmpdir)/jpeg-$(VERSION)

install: $(SOURCE)
	cd $(SOURCE) ; ./configure --prefix=$(build_prefix) --disable-static && make install-libLTLIBRARIES
ifeq ($(OS),Darwin)
	libjpeg=libjpeg.$(LIBJPEG_VER).dylib; \
	install_name_tool -id "@rpath/$$libjpeg" $(shlibdir)/$$libjpeg
endif

app-install:;
	$(RSYNC) $(shlibdir)/libjpeg.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
