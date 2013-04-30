FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.9.0
DISTRIBUTION = glew-$(VERSION).tgz
SOURCE = $(tmpdir)/glew-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

MAKE_ARGS = CC='$(QUOTE_CC)' OPT='$(OPT)' GLEW_DEST='$(build_prefix)' LIBDIR='$(libdir)'

all:

install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifeq ($(OS),Windows)
	cp Makefile.msvc $(SOURCE)/Makefile
endif

clean:
	rm -rf $(SOURCE)
