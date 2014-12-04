FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.8.10.2
ifdef WIN32
DISTRIBUTION = cmake-$(VERSION)-win32-x86.zip
SOURCE = $(tmpdir)/cmake-$(VERSION)-win32-x86
else
DISTRIBUTION = cmake-$(VERSION).tar.gz
SOURCE = $(tmpdir)/cmake-$(VERSION)
endif

PATCHES = $(wildcard cmake-$(VERSION)-*.patch)

all:

install: $(SOURCE)
ifdef WIN32
	cd $(SOURCE)/bin && chmod 755 *.exe *.dll
	cd $(SOURCE) && cp -ruv * $(build_prefix)
else
	cd $(SOURCE) && \
		./bootstrap --prefix=$(build_prefix) && \
		 $(MAKE) install
endif

app-install:;

$(SOURCE): $(DISTRIBUTION)
ifdef WIN32
	unzip -d $(tmpdir) -q $(DISTRIBUTION)
else
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
endif

clean:
	rm -rf $(SOURCE)
