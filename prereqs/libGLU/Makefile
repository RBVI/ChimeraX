PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 9.0.2
DISTRIBUTION = glu-glu-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/glu-glu-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

CONFIG_OPTS = --prefix '$(build_prefix)'

all:

install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) install

app-install:
	cp -av $(includedir)/libGLU.h $(app_includedir)
	cp -av $(libdir)/libGLU $(app_libdir)

$(SOURCE): $(DISTRIBUTION)
	tar jxf $(DISTRIBUTION) -C $(tmpdir)
	# remove uses of libGL
	cd $(SOURCE) && sed -i -e '/mipmap/d' -e '/project/d' -e '/quad/d' -e '/LIBADD/d' Makefile.am
	cd $(SOURCE) && ./autogen.sh $(CONFIG_OPTS)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/libGLU/$(DISTRIBUTION)

clean:
	rm -rf $(SOURCE)
