FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 8.36
LIBPCRE_VER = 1
LIBPCRECPP_VER = 0
LIBPCREPOSIX_VER = 0
DISTRIBUTION = pcre-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/pcre-$(VERSION)

ifdef UNIX
QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC)' CXX='$(QUOTE_CXX)'
CONF_ARGS = --prefix=$(build_prefix) --enable-utf --enable-unicode-properties --disable-static

ifdef DEBUG
MAKE_ARGS = CFLAGS="$(OPT)" CXXFLAGS="$(OPT)"
ifeq ($(OS),Linux)
CONF_ARGS += --enable-valgrind
endif
endif

endif # UNIX

ifeq ($(OS),Windows)
CMAKE_OPTS = -DCMAKE_INSTALL_PREFIX=$(build_prefix) \
	-DCMAKE_PREFIX_PATH=$(build_prefix) \
	-DBUILD_SHARED_LIBS=1 \
	-DPCRE_SUPPORT_UTF=1 -DPCRE_SUPPORT_UNICODE_PROPERTIES=1
ifdef DEBUG
CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Debug
else
CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Release
endif

ifeq ($(OSARCH),Windows64)
CMAKE_OPTS += -G 'Visual Studio 9 2008 Win64'
CONFIG = Release
PLATFORM = x64
else
CMAKE_OPTS += -G 'Visual Studio 9 2008'
CONFIG = Release
PLATFORM = Win32
endif
CONFIG_PLATFORM = '$(CONFIG)|$(PLATFORM)'

endif # OS == Windows

all:

install: $(SOURCE)
ifeq ($(OS),Windows)
	cd $(SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcre
	cd $(SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcreposix
	cd $(SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcrecpp
	$(RSYNC) $(SOURCE)/build/Release/*.$(SHLIB_EXT) $(shlibdir)
	$(RSYNC) $(SOURCE)/build/Release/*.$(LIB_EXT) $(libdir)
	$(RSYNC) $(SOURCE)/pcre{cpp,posix,_scanner}.h $(includedir)
	$(RSYNC) $(SOURCE)/build/pcre{,cpparg,_stringpiece}.h $(includedir)
else
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) && $(MAKE) $(MAKE_ARGS) install
ifeq ($(OS),Linux)
	cd $(bindir); chrpath -r '$$ORIGIN/../lib' pcregrep pcretest
endif
ifeq ($(OS),Darwin)
	cd $(shlibdir); \
	libpcre=libpcre.$(LIBPCRE_VER).dylib; \
	libpcrecpp=libpcrecpp.$(LIBPCRECPP_VER).dylib; \
	libpcreposix=libpcreposix.$(LIBPCREPOSIX_VER).dylib; \
	install_name_tool -id "@rpath/$$libpcre" $$libpcre; \
	install_name_tool -id "@rpath/$$libpcrecpp" $$libpcrecpp; \
	install_name_tool -id "@rpath/$$libpcreposix" $$libpcreposix; \
	install_name_tool -change $(shlibdir)/$$libpcre "@rpath/$$libpcre" $$libpcrecpp ; \
	install_name_tool -change $(shlibdir)/$$libpcre "@rpath/$$libpcre" $$libpcreposix ; \
	cd $(bindir); for f in pcregrep pcretest; do \
		install_name_tool -add_rpath '@executable_path/../lib' $$f ; \
		install_name_tool -change $(shlibdir)/$$libpcre "@rpath/$$libpcre" $$f ; \
		install_name_tool -change $(shlibdir)/$$libpcreposix "@rpath/$$libpcreposix" $$f ; \
	done
endif
endif

app-install:;

$(SOURCE):
	tar jxf $(DISTRIBUTION) -C $(tmpdir)
ifdef UNIX
	cd $(SOURCE) && $(ENV_CONFIGURE) ./configure $(CONF_ARGS)
endif
ifeq ($(OS),Windows)
	mkdir $(SOURCE)/build && cd $(SOURCE)/build && cmake $(CMAKE_OPTS) ..
endif

clean:
	rm -rf $(SOURCE)
