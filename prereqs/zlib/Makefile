PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.2.11
LIBZ_VER = 1
DISTRIBUTION = zlib-$(VERSION).tar.gz
SOURCE = $(tmpdir)/zlib-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

ENV_CONFIGURE = env prefix='$(build_prefix)' CC='$(QUOTE_CC)' CFLAGS='$(OPT)'

all:

install: $(SOURCE)
ifndef WIN32
	cd $(SOURCE) && $(MAKE) install
ifeq ($(OS),Darwin)
	libz=libz.$(LIBZ_VER).dylib; \
	install_name_tool -id "@rpath/$$libz" $(shlibdir)/$$libz
endif
else
	cd $(SOURCE) && env MAKEFLAGS= MFLAGS= nmake /nologo /f win32/Makefile.msc
	cd $(SOURCE) && \
		$(RSYNC) zdll.lib $(libdir) ; \
		$(RSYNC) zdll1.lib $(shlibdir) ; \
		$(RSYNC) zconf.h zlib.h $(includedir)
endif

app-install:
	$(RSYNC) $(includedir)/zlib.h $(app_includedir)
	$(RSYNC) $(shlibdir)/libz.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifndef WIN32
	cd $(SOURCE) && $(ENV_CONFIGURE) ./configure
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/zlib/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/lib

clean:
	rm -rf $(SOURCE)
