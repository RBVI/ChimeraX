FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

XSLT_VERSION = 1.1.28
XSLT_PACKAGE = libxslt-$(XSLT_VERSION).tar.gz
XSLT_SOURCE = $(tmpdir)/libxslt-$(XSLT_VERSION)

CONFIG_ENV = env LDFLAGS='-L$(libdir)'
CONFIG_OPTS = --prefix='$(build_prefix)' --disable-static --with-python

all:

install: $(XSLT_SOURCE)
	cd $(XSLT_SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install

$(XSLT_SOURCE): $(XSLT_PACKAGE)
	tar zxf $(XSLT_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(XSLT_SOURCE)
