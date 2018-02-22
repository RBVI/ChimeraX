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

app-install:
	$(RSYNC) $(bindir)/chrpath$(PROG_EXT) $(app_bindir)

clean:
	rm -rf $(SOURCE)

$(PROG): $(SOURCE)
	cd $(SOURCE) && ./configure $(CONFIG_OPTS) && $(MAKE)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
