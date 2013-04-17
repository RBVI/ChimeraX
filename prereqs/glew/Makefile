FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

GLEW_VERSION = 1.9.0
GLEW_PACKAGE = glew-$(GLEW_VERSION).tgz
GLEW_SOURCE = $(tmpdir)/glew-$(GLEW_VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

MAKE_ARGS = CC='$(QUOTE_CC)' OPT='$(OPT)' GLEW_DEST='$(build_prefix)' LIBDIR='$(libdir)'

all:

install: $(GLEW_SOURCE)
	cd $(GLEW_SOURCE) && $(MAKE) $(MAKE_ARGS) install

$(GLEW_SOURCE):
	tar zxf $(GLEW_PACKAGE) -C $(tmpdir)
ifeq ($(OS),Windows)
	cp Makefile.msvc $(GLEW_SOURCE)/Makefile
endif

clean:
	rm -rf $(GLEW_SOURCE)
