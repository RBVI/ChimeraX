FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

NUMPY_VERSION = 1.7.0
NUMPY_PACKAGE = numpy-$(NUMPY_VERSION).tar.gz
NUMPY_SOURCE = $(tmpdir)/numpy-$(NUMPY_VERSION)

SETUP_BUILD = build --fcompiler=none
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

VER	:= $(shell expr $(NUMPY_VERSION) : '\(.*\..*\..*\)\.')
ifeq ($(VER),)
VER	= $(NUMPY_VERSION)
endif

PATCHES = $(wildcard numpy-$(VER)-fcompiler.patch)

ifeq ($(OSARCH),Linux64)
PATCHES += $(wildcard numpy-$(VER)-linux64-syslibs.patch)
endif

all:

install: $(NUMPY_SOURCE)
	cd $(NUMPY_SOURCE) && python setup.py $(SETUP_BUILD) install

$(NUMPY_SOURCE):
	tar zxf $(NUMPY_PACKAGE) -C $(tmpdir)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		(cd $(NUMPY_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(NUMPY_SOURCE)
