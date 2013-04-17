FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

XML2_VERSION = 2.9.0
XML2_PACKAGE = libxml2-sources-$(XML2_VERSION).tar.gz
XML2_SOURCE = $(tmpdir)/libxml2-$(XML2_VERSION)

PATCHES	= $(wildcard libxml2-$(XML2_VERSION)-*.patch)

CONFIG_ENV = env LDFLAGS='-L$(libdir)'
CONFIG_OPTS = --prefix='$(build_prefix)' --disable-static --with-python --with-zlib

all:

install: $(XML2_SOURCE)
	cd $(XML2_SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install

$(XML2_SOURCE): $(XML2_PACKAGE)
	tar zxf $(XML2_PACKAGE) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(XML2_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(XML2_SOURCE)
