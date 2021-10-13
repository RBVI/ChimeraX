PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 19.0.8
DISTRIBUTION = mesa-$(VERSION).tar.xz
SOURCE = $(tmpdir)/mesa-$(VERSION)

ifdef CENTOS_DIST
ifeq ($(shell expr $(CENTOS_DIST) '<=' 7),1)
MESA_ENV += LLVM_CONFIG=mesa-private-llvm-config-64
else
$(error "Mesa is not needed for CentOS $(CENTOS)")
endif
endif

PATCHES	= $(wildcard mesa-$(VERSION)-$(OS).patch)

CONFIG_OPTS = --prefix=$(build_prefix) \
	--enable-gallium-osmesa \
	--with-gallium-drivers="swrast" \
	--enable-llvm=yes \
	--with-platforms= \
	--with-dri-drivers= \
	--disable-dri \
	--disable-egl \
	--disable-gbm \
	--disable-gles1 \
	--disable-gles2 \
	--disable-glx \
	--disable-va \
	--disable-vdpau \
	--disable-xvmc \
	--disable-shared-glapi \
	--enable-autotools

ifneq (,$(wildcard $(tmpdir)/llvm-*))
#if using our own llvm:
CONFIG_OPTS += --with-llvm-prefix=$(build_prefix) \
	--disable-llvm-shared-libs
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  $(MESA_ENV) ./configure $(CONFIG_OPTS) && $(MAKE) install

app-install:;
#	$(RSYNC) $(shlibdir)/libOSMesa.*$(SHLIB_EXT)* $(app_shlibdir)

$(SOURCE) source: $(DISTRIBUTION)
	tar -xf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/mesa/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/mesa

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
