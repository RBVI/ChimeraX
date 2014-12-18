FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.1.28
DISTRIBUTION = libxslt-$(VERSION).tar.gz
SOURCE = $(tmpdir)/libxslt-$(VERSION)

CONFIG_ENV = env LDFLAGS='-L$(libdir)'
CONFIG_OPTS = --prefix='$(build_prefix)' --disable-static --with-python

all:

install: $(SOURCE)
	cd $(SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
