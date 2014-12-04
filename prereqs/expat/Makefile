FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.1.0
DISTRIBUTION = expat-$(VERSION).tar.gz
SOURCE = $(tmpdir)/expat-$(VERSION)
SHLIB_VER = 1

PATCHES = $(wildcard expat-$(VER)*.patch)

ifdef WIN32
RSYNC = cp -uv
endif

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# Make Python use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC)' CXX='$(QUOTE_CXX)' LDFLAGS="$(TARGET_ARCH) $(LDFLAGS)"

HEADERS = expat.h expat_external.h

all:

install: $(SOURCE)
ifeq ($(LDFLAGS),)
	cd $(SOURCE) && $(MAKE) buildlib
else
	cd $(SOURCE) && $(MAKE) LDFLAGS="$(LDFLAGS)" buildlib
endif
	cd $(SOURCE) && $(MAKE) installlib
ifeq ($(OS),Darwin)
	install_name_tool -id \
	  @executable_path/../lib/libexpat.$(SHLIBVER).$(SHLIB_EXT) \
	  $(libdir)/libexpat.$(SHLIB_EXT)
endif

app-install:;
	$(RSYNC) $(shlibdir)/libexpat.$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for i in $(PATCHES) ; do \
		(cd $(SOURCE) && patch -f -p0) < $$i; \
	done
endif
ifeq ($(OS),Windows)
ifeq ($(WIN32),mingw)
	cd $(SOURCE) && \
	  $(ENV_CONFIGURE) ./configure --prefix=$(build_prefix) \
					--build=i686-pc-mingw32
	(cd $(SOURCE) && patch -f -p0) < libtool.patch
endif
ifeq ($(WIN32),msvc)
	sed 's;^INSTDIR.*;INSTDIR=$(build_prefix);' < Makefile.vc > $(SOURCE)/Makefile
endif
else
	cd $(SOURCE) && \
	  $(ENV_CONFIGURE) ./configure --prefix=$(build_prefix)
endif

clean:
	rm -rf $(SOURCE)
