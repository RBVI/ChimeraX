FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

OPENSSL_VERSION = 1.0.1e
OPENSSL_PACKAGE = openssl-$(OPENSSL_VERSION).tar.gz
OPENSSL_SOURCE = $(tmpdir)/openssl-$(OPENSSL_VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# Make compilation use our compiler options
MAKE_ARGS = CC='$(QUOTE_CC)' CXX='$(QUOTE_CXX)'
ENV_CONFIGURE = env $(MAKE_ARGS)

PATCHES	= $(wildcard openssl-$(OPENSSL_VERSION)-*.patch) \
		$(wildcard openssl-$(OPENSSL_VERSION)_$(OS).patch) \
		$(wildcard openssl-$(OPENSSL_VERSION)_$(OSARCH).patch)
POSTCONFIG_PATCHES	= $(wildcard postconfig-$(OPENSSL_VERSION)-*.patch) \
		$(wildcard postconfig-$(OPENSSL_VERSION)_$(OS).patch) \
		$(wildcard postconfig-$(OPENSSL_VERSION)_$(OSARCH).patch)

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

all:

install: $(OPENSSL_SOURCE)
ifndef WIN32
	cd $(OPENSSL_SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl
	cd $(OPENSSL_SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl build-shared
	cd $(OPENSSL_SOURCE) && $(MAKE) $(MAKE_ARGS) -j1 PERL=perl install
else
	cd $(OPENSSL_SOURCE) && env MFLAGS= MAKEFLAGS= nmake /nologo /f ms/ntdll.mak
	$(RSYNC) $(OPENSSL_SOURCE)/out32dll/libeay32.dll $(shlibdir)
	$(RSYNC) $(OPENSSL_SOURCE)/out32dll/libeay32.lib $(libdir)/libcrypto.lib
	$(RSYNC) --copy-links $(OPENSSL_SOURCE)/inc32/openssl $(includedir)
endif
ifeq ($(OS),Darwin)
	cd $(shlibdir) && \
	SHLIB_VERSION=`awk '$$2 == "SHLIB_VERSION_NUMBER" { print $$3 }' $(includedir)/openssl/opensslv.h | sed s/\"//g` && \
	chmod 644 \
	    libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    libssl.$$SHLIB_VERSION.$(SHLIB_EXT) && \
	install_name_tool -id \
	    @executable_path/../lib/libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) && \
	install_name_tool -id \
	    @executable_path/../lib/libssl.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    libssl.$$SHLIB_VERSION.$(SHLIB_EXT) && \
	install_name_tool -change \
	    libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    @executable_path/../lib/libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    libssl.$$SHLIB_VERSION.$(SHLIB_EXT) && \
	chmod 444 \
	    libcrypto.$$SHLIB_VERSION.$(SHLIB_EXT) \
	    libssl.$$SHLIB_VERSION.$(SHLIB_EXT)
endif

$(OPENSSL_SOURCE):
	tar zxf $(OPENSSL_PACKAGE) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(OPENSSL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif
	cd $(OPENSSL_SOURCE) && \
	  $(ENV_CONFIGURE) $(CONFIG) --prefix=$(build_prefix) $(TARGET) shared
ifdef WIN32
	cd $(OPENSSL_SOURCE) && cmd /c $(BAT)
	rm -f $(OPENSSL_SOURCE)/NUL
endif
ifneq (,$(POSTCONFIG_PATCHES))
	for p in $(POSTCONFIG_PATCHES); do \
		(cd $(OPENSSL_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(OPENSSL_SOURCE)
