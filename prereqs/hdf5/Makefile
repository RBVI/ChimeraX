FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.8.14
LIBHDF5_VER = 9
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
ifeq ($(OS),Darwin)
	-cd $(shlibdir); \
	libhdf5=libhdf5.$(LIBHDF5_VER).dylib; \
	libhdf5_hl=libhdf5_hl.$(LIBHDF5_VER).dylib; \
	install_name_tool -id "@rpath/$$libhdf5" $$libhdf5; \
	install_name_tool -id "@rpath/$$libhdf5_hl" $$libhdf5_hl; \
	install_name_tool -change $(shlibdir)/$$libhdf5 "@rpath/$$libhdf5" $$libhdf5_hl ; \
	cd $(bindir); for f in *h5*; do \
		install_name_tool -add_rpath '@executable_path/../lib' $$f ; \
		install_name_tool -change $(shlibdir)/$$libhdf5 "@rpath/$$libhdf5" $$f ; \
		install_name_tool -change $(shlibdir)/$$libhdf5_hl "@rpath/$$libhdf5_hl" $$f ; \
	done
endif

app-install:;
	$(RSYNC) $(includedir)/hdf5*.h $(includedir)/H5* $(app_includedir)
	$(RSYNC) $(shlibdir)/libhdf5.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	tar jxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
