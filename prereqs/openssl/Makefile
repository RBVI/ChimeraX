PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.1.1n
SHLIB_VERSION=$(shell awk '$$3 == "SHLIB_VERSION_NUMBER" { print $$4 }' $(SOURCE)/include/openssl/opensslv.h | sed s/\"//g)
DISTRIBUTION = openssl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/openssl-$(VERSION)
OPENSSL_OLD_HEADERS = openssl-include-0.9.8.tar.gz

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

# Make compilation use our compiler options
MAKE_ARGS = CC='$(QUOTE_CC)'
ENV_CONFIGURE = env $(MAKE_ARGS)

PATCHES	= $(wildcard openssl-$(VERSION)-*.patch) \
		$(wildcard openssl-$(VERSION)_$(OS).patch) \
		$(wildcard openssl-$(VERSION)_$(OSARCH).patch)
POSTCONFIG_PATCHES	= $(wildcard postconfig-$(VERSION)-*.patch) \
		$(wildcard postconfig-$(VERSION)_$(OS).patch) \
		$(wildcard postconfig-$(VERSION)_$(OSARCH).patch)

CONFIG	= ./config
ifeq ($(OSARCH),Linux)
CONFIG	= ./Configure
TARGET	= linux-elf -Wa,--noexecstack
else ifeq ($(OSARCH),Linux64)
CONFIG	= ./Configure
TARGET	= linux-x86_64 -Wa,--noexecstack
else ifeq ($(OSARCH),Windows)
CONFIG	= perl Configure
TARGET	= VC-WIN32 -DUNICODE -D_UNICODE
BAT	= 'ms\do_ms.bat'
else ifeq ($(OSARCH),Windows64)
CONFIG	= perl Configure
TARGET	= VC-WIN64A
BAT	= 'ms\do_win64a.bat'
else ifeq ($(OSARCH),DarwinIntel)
CONFIG	= ./Configure
TARGET	= darwin-i386-cc
ifdef DEBUG
TARGET	= debug-darwin-i386-cc
endif
else ifeq ($(OSARCH),DarwinIntel64)
CONFIG	= ./Configure
TARGET	= darwin64-x86_64-cc
endif

ifeq ($(OS),Darwin)
# Using old headers no longer works because PyPI requires TLS 1.2+
# and OpenSSL 0.9.8 does not speak that.  Instead, we will compile
# a more modern OpenSSL and point it to use the certificates file 
# from the "certifi" package, which can be updated from PyPI.
#
# USE_OLD_HEADERS=1
# OpenSSL headers for 0.9.8 were copied from
# /usr/local/src/MacOSX.sdks/MacOSX10.10.sdk/usr/include/openssl
# from El Capitan
endif

all:

ifdef WIN32
install: $(SOURCE)
	cd $(SOURCE) && env MFLAGS= MAKEFLAGS= nmake /nologo /f ms/ntdll.mak
	$(RSYNC) $(SOURCE)/out32dll/libeay32.dll $(shlibdir)
	$(RSYNC) $(SOURCE)/out32dll/libeay32.lib $(libdir)/libcrypto.lib
	$(RSYNC) --copy-links $(SOURCE)/inc32/openssl $(includedir)
endif

ifeq ($(OS),Linux)
install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl install_sw
	#chrpath -r '$$ORIGIN/../lib' $(bindir)/openssl
endif

ifeq ($(OS),Darwin)
ifdef USE_OLD_HEADERS
install: $(OPENSSL_OLD_HEADERS)
	tar zxf $(OPENSSL_OLD_HEADERS) -C $(includedir)
else
install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl install_sw
	cd $(shlibdir) && \
	chmod ug+w \
	    libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
	    libssl.$(SHLIB_VERSION).$(SHLIB_EXT); \
	install_name_tool -id \
	    @rpath/libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
	    libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) ; \
	install_name_tool -id \
	    @rpath/libssl.$(SHLIB_VERSION).$(SHLIB_EXT) \
	    libssl.$(SHLIB_VERSION).$(SHLIB_EXT) ; \
	install_name_tool -change \
	    $(shlibdir)/libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
	    @rpath/libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
	    libssl.$(SHLIB_VERSION).$(SHLIB_EXT)
	cd $(libdir)/engines*; chmod ug+w *.dylib; for f in *.dylib; do \
		install_name_tool -change \
			$(shlibdir)/libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
			@rpath/libcrypto.$(SHLIB_VERSION).$(SHLIB_EXT) \
			$$f; \
	done
endif
endif

app-install:
ifndef WIN32
ifndef USE_OLD_HEADERS
	$(RSYNC) $(shlibdir)/libcrypto.*$(SHLIB_EXT)* \
		$(shlibdir)/libssl.*$(SHLIB_EXT)* $(app_shlibdir)
	# do we need $(shlibdir)/engines/?
endif
endif

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
	cd $(SOURCE) && \
	  $(ENV_CONFIGURE) $(CONFIG) --prefix=$(build_prefix) $(TARGET) shared no-tests
ifdef WIN32
	cd $(SOURCE) && cmd /c $(BAT)
	rm -f $(SOURCE)/NUL
endif
ifneq (,$(POSTCONFIG_PATCHES))
	for p in $(POSTCONFIG_PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openssl/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/openssl

$(OPENSSL_OLD_HEADERS):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/openssl/$(OPENSSL_OLD_HEADERS)

clean:
	rm -rf $(SOURCE)
