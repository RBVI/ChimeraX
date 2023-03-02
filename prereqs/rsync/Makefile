PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.2.7
DISTRIBUTION = rsync-$(VERSION).tar.gz
SOURCE = $(tmpdir)/rsync-$(VERSION)

QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))

CONFIG_OPTS = --prefix '$(build_prefix)' --disable-xxhash --disable-zstd --disable-lz4

all:

install: $(SOURCE)
	cd $(SOURCE) && $(MAKE) install

app-install:
	cp $(bindir)/rsync $(app_bindir)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
	cd $(SOURCE) && ./configure $(CONFIG_OPTS)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/rsync/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/lib

clean:
	rm -rf $(SOURCE)
