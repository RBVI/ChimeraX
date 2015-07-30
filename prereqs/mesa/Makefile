FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 10.6.3
DISTRIBUTION = mesa-$(VERSION).tar.gz
SOURCE = $(tmpdir)/mesa-$(VERSION)

ifeq ($(OS),Darwin)
PATCHES	= mesa-10.6.3-darwin.patch
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  PKG_CONFIG=$(bin_dir)/pkg-config \
	  ./configure --prefix=$(build_prefix) \
		--enable-gallium-osmesa \
		--with-gallium-drivers="swrast" \
		--enable-gallium-llvm=yes \
		--with-llvm-prefix=$(build_prefix) \
		--disable-llvm-shared-libs \
		--disable-dri \
		--disable-glx \
		--disable-egl \
		--disable-shared-glapi && \
	  make install

app-install:;
	$(RSYNC) $(shlibdir)/libOSMesa.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
