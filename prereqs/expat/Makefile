FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

EXPAT_VERSION = 2.1.0
EXPAT_PACKAGE = expat-$(EXPAT_VERSION).tar.gz
EXPAT_SOURCE = $(tmpdir)/expat-$(EXPAT_VERSION)
EXPAT_SHLIB_VER = 1

PATCHES = $(wildcard expat-$(EXPAT_VER)*.patch)

ifdef WIN32
RSYNC = cp -uv
endif

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# Make Python use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC)' CXX='$(QUOTE_CXX)' LDFLAGS="$(TARGET_ARCH) $(LDFLAGS)"

EXPAT_HEADERS = expat.h expat_external.h

all:

install: $(EXPAT_SOURCE)
ifeq ($(LDFLAGS),)
	cd $(EXPAT_SOURCE) && $(MAKE) buildlib
else
	cd $(EXPAT_SOURCE) && $(MAKE) LDFLAGS="$(LDFLAGS)" buildlib
endif
	cd $(EXPAT_SOURCE) && $(MAKE) installlib
ifeq ($(OS),Darwin)
	install_name_tool -id \
	  @executable_path/../lib/libexpat.$(EXPAT_SHLIBVER).$(SHLIB_EXT) \
	  $(libdir)/libexpat.$(SHLIB_EXT)
endif

$(EXPAT_SOURCE):
	tar zxf $(EXPAT_PACKAGE) -C $(tmpdir)
ifneq (,$(PATCHES))
	for i in $(PATCHES) ; do \
		(cd $(EXPAT_SOURCE) && patch -f -p0) < $$i; \
	done
endif
ifeq ($(OS),Windows)
ifeq ($(WIN32),mingw)
	cd $(EXPAT_SOURCE) && \
	  $(ENV_CONFIGURE) ./configure --prefix=$(build_prefix) \
					--build=i686-pc-mingw32
	(cd $(EXPAT_SOURCE) && patch -f -p0) < libtool.patch
endif
ifeq ($(WIN32),msvc)
	sed 's;^INSTDIR.*;INSTDIR=$(build_prefix);' < Makefile.vc > $(EXPAT_SOURCE)/Makefile
endif
else
	cd $(EXPAT_SOURCE) && \
	  $(ENV_CONFIGURE) ./configure --prefix=$(build_prefix)
endif

clean:
	rm -rf $(EXPAT_SOURCE)
