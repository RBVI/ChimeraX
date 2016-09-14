PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 12.0.2
DISTRIBUTION = mesa-$(VERSION).tar.xz
SOURCE = $(tmpdir)/mesa-$(VERSION)

ifeq ($(OS),Darwin)
MESA_ENV = PKG_CONFIG=$(bindir)/pkg-config
endif

#PATCHES	= $(wildcard mesa-$(VERSION)-$(OS).patch)

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  $(MESA_ENV) ./configure --prefix=$(build_prefix) \
		--enable-gallium-osmesa \
		--with-gallium-drivers="swrast" \
		--enable-gallium-llvm=yes \
		--with-llvm-prefix=$(build_prefix) \
		--disable-llvm-shared-libs \
		--disable-dri \
		--disable-glx \
		--disable-egl \
		--disable-gles1 \
		--disable-gles2 \
		--disable-shared-glapi && \
	  $(MAKE) install

app-install:;
	$(RSYNC) $(shlibdir)/libOSMesa.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE):
	$(bindir)/7za e -so $(DISTRIBUTION) | tar xf - -C $(tmpdir)
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
