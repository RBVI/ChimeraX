FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.8.14
DISTRIBUTION = hdf5-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/hdf5-$(VERSION)

CONFIG_ENV = env
CONFIG_OPTS = --prefix='$(build_prefix)' --with-zlib='$(includedir),$(libdir)'

all:

install: $(SOURCE)
	cd $(SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install
ifeq ($(OS),Linux)
	cd $(bindir); for i in *h5*; do \
		chrpath -r '$$ORIGIN/../lib' $$i; \
	done
endif

$(SOURCE):
	tar jxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
