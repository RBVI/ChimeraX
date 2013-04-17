FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

CMAKE_VERSION = 2.8.10.2
ifdef WIN32
CMAKE_PACKAGE = cmake-$(CMAKE_VERSION)-win32-x86.zip
CMAKE_SOURCE = $(tmpdir)/cmake-$(CMAKE_VERSION)-win32-x86
else
CMAKE_PACKAGE = cmake-$(CMAKE_VERSION).tar.gz
CMAKE_SOURCE = $(tmpdir)/cmake-$(CMAKE_VERSION)
endif

PATCHES = $(wildcard cmake-$(CMAKE_VERSION)-*.patch)

all:

install: $(CMAKE_SOURCE)
ifdef WIN32
	cd $(CMAKE_SOURCE)/bin && chmod 755 *.exe *.dll
	cd $(CMAKE_SOURCE) && cp -ruv * $(build_prefix)
else
	cd $(CMAKE_SOURCE) && \
		./bootstrap --prefix=$(build_prefix) && \
		 $(MAKE) install
endif

$(CMAKE_SOURCE): $(CMAKE_PACKAGE)
ifdef WIN32
	unzip -d $(tmpdir) -q $(CMAKE_PACKAGE)
else
	tar zxf $(CMAKE_PACKAGE) -C $(tmpdir)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		(cd $(CMAKE_SOURCE) && patch -f -p0) < $$p ; \
	done
endif
endif

clean:
	rm -rf $(CMAKE_SOURCE)
