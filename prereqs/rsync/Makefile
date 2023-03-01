PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.2.7
DISTRIBUTION = rsync-$(VERSION).tar.gz
PATCHES_DISTRIBUTION = rsync-patches-$(VERSION).tar.gz
SOURCE = $(tmpdir)/rsync-$(VERSION)
CONFIG_OPTS = --disable-xxhash --disable-zstd --disable-lz4

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

ENV_CONFIGURE = env prefix='$(build_prefix)'

all:

install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) install

app-install:
	cp $(bindedir)/rsync $(app_bindir)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
	tar zxf $(PATCHES_DISTRIBUTION) -C $(tmpdir)
	-cd $(SOURCE) && \
	for p in patches/*; do \
		patch -f -p1 < $$p; \
	done
	cd $(SOURCE) && $(ENV_CONFIGURE) ./configure $(CONFIG_OPTS)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/rsync/$(DISTRIBUTION)
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/rsync/$(PATCHES_DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/lib

clean:
	rm -rf $(SOURCE)
