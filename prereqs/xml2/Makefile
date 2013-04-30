FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.9.0
DISTRIBUTION = libxml2-sources-$(VERSION).tar.gz
SOURCE = $(tmpdir)/libxml2-$(VERSION)

PATCHES	= $(wildcard libxml2-$(VERSION)-*.patch)

CONFIG_ENV = env LDFLAGS='-L$(libdir)'
CONFIG_OPTS = --prefix='$(build_prefix)' --disable-static --with-python --with-zlib

all:

install: $(SOURCE)
	cd $(SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
