PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.4.6
DISTRIBUTION = libtool-$(VERSION).tar.gz
SOURCE = $(tmpdir)/libtool-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
ENV_CONFIGURE = env CC='$(QUOTE_CC)' LDFLAGS="$(TARGET_ARCH) $(LDFLAGS)"

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  $(ENV_CONFIGURE) ./configure --prefix=$(build_prefix)
	cd $(SOURCE) && $(MAKE) $(MAKE_ARGS) install

app-install:;

source $(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
