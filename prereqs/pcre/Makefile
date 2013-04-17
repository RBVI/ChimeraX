FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PCRE_VERSION = 8.32
PCRE_PACKAGE = pcre-$(PCRE_VERSION).tar.bz2
PCRE_SOURCE = $(tmpdir)/pcre-$(PCRE_VERSION)

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

install: $(PCRE_SOURCE)
ifeq ($(OS),Windows)
	cd $(PCRE_SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcre
	cd $(PCRE_SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcreposix
	cd $(PCRE_SOURCE)/build && devenv.com PCRE.sln /build $(CONFIG_PLATFORM) /project pcrecpp
	$(RSYNC) $(PCRE_SOURCE)/build/Release/*.$(SHLIB_EXT) $(shlibdir)
	$(RSYNC) $(PCRE_SOURCE)/build/Release/*.$(LIB_EXT) $(libdir)
	$(RSYNC) $(PCRE_SOURCE)/pcre{cpp,posix,_scanner}.h $(includedir)
	$(RSYNC) $(PCRE_SOURCE)/build/pcre{,cpparg,_stringpiece}.h $(includedir)
else
	cd $(PCRE_SOURCE) && $(MAKE) $(MAKE_ARGS) && $(MAKE) $(MAKE_ARGS) install
endif

$(PCRE_SOURCE):
	tar jxf $(PCRE_PACKAGE) -C $(tmpdir)
ifdef UNIX
	cd $(PCRE_SOURCE) && $(ENV_CONFIGURE) ./configure $(CONF_ARGS)
endif
ifeq ($(OS),Windows)
	mkdir $(PCRE_SOURCE)/build && cd $(PCRE_SOURCE)/build && cmake $(CMAKE_OPTS) ..
endif

clean:
	rm -rf $(PCRE_SOURCE)
