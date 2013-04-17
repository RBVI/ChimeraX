FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ZLIB_VERSION = 1.2.7
ZLIB_PACKAGE = zlib-$(ZLIB_VERSION).tar.gz
ZLIB_SOURCE = $(tmpdir)/zlib-$(ZLIB_VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

ENV_CONFIGURE = env prefix='$(build_prefix)' CC='$(QUOTE_CC)' CFLAGS='$(OPT)'

all:

install: $(ZLIB_SOURCE)
ifndef WIN32
	cd $(ZLIB_SOURCE) && $(MAKE) install
else
	cd $(ZLIB_SOURCE) && env MAKEFLAGS= MFLAGS= nmake /nologo /f win32/Makefile.msc
	cd $(ZLIB_SOURCE) && \
		$(RSYNC) zdll.lib $(libdir) ; \
		$(RSYNC) zdll1.lib $(shlibdir) ; \
		$(RSYNC) zconf.h zlib.h $(includedir)
endif

$(ZLIB_SOURCE):
	tar zxf $(ZLIB_PACKAGE) -C $(tmpdir)
ifndef WIN32
	cd $(ZLIB_SOURCE) && $(ENV_CONFIGURE) ./configure
endif

clean:
	rm -rf $(ZLIB_SOURCE)
