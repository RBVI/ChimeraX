PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.16

# Windows zip file is needed on all platforms because it has all of the
# sublicences in it that are needed for the documentation.

DISTRIBUTION = chrpath-$(VERSION).tar.gz
SOURCE = $(tmpdir)/chrpath-$(VERSION)
PROG = $(SOURCE)/chrpath

CONFIG_OPTS	=  --prefix=$(build_prefix)

all: $(PROG)

install: $(PROG)
	$(MAKE) -C $(SOURCE) install

app-install:;

clean:
	rm -rf $(SOURCE)

$(PROG): $(SOURCE)
	cd $(SOURCE) && ./configure $(CONFIG_OPTS) && $(MAKE)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/chrpath/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/chrpath
