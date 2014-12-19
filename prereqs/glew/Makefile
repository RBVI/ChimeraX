FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.10.0
LIBGLEW_VER = 1.10.0
DISTRIBUTION = glew-$(VERSION).tgz
SOURCE = $(tmpdir)/glew-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

MAKE_ARGS = CC='$(QUOTE_CC)' OPT='$(OPT)' GLEW_DEST='$(build_prefix)' LIBDIR='$(libdir)'

all:

install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) install
ifeq ($(OS),Darwin)
	libglew=libGLEW.$(LIBGLEW_VER).dylib; \
	install_name_tool -id "@rpath/$$libglew" $(shlibdir)/$$libglew
endif

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifeq ($(OS),Windows)
	cp Makefile.msvc $(SOURCE)/Makefile
endif

clean:
	rm -rf $(SOURCE)
