PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ifdef WIN32
$(error Windows uses binary version distributed with tables package)
endif
VERSION = 1.10.4
LIBHDF5_VER = 103
DISTRIBUTION = hdf5-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/hdf5-$(VERSION)

CONFIG_ENV = env
CONFIG_OPTS = --prefix='$(build_prefix)' --with-zlib='$(includedir),$(libdir)' \
	      --enable-threadsafe --disable-static --disable-hl

all:

install: $(SOURCE)
	cd $(SOURCE) && $(CONFIG_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install
ifeq ($(OS),Linux)
	cd $(libdir) && chrpath -d libhdf5*.so
	cd $(bindir) && for i in *h5*; do \
		chrpath -r '$$ORIGIN/../lib' $$i; \
	done
endif
ifeq ($(OS),Darwin)
	-cd $(shlibdir); \
	libhdf5=libhdf5.$(LIBHDF5_VER).dylib; \
	install_name_tool -id "@rpath/$$libhdf5" $$libhdf5; \
	cd $(bindir); for f in *h5*; do \
		install_name_tool -add_rpath '@executable_path/../lib' $$f ; \
		install_name_tool -change $(shlibdir)/$$libhdf5 "@rpath/$$libhdf5" $$f ; \
	done
endif

app-install:;
	$(RSYNC) $(includedir)/hdf5*.h $(includedir)/H5* $(app_includedir)
	$(RSYNC) $(shlibdir)/libhdf5*.$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	tar jxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
