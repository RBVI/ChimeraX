FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.9.1
DISTRIBUTION = numpy-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numpy-$(VERSION)

SETUP_BUILD = build --fcompiler=none
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

VER	:= $(shell expr $(VERSION) : '\(.*\..*\..*\)\.')
ifeq ($(VER),)
VER	= $(VERSION)
endif

PATCHES = $(wildcard numpy-$(VER)-fcompiler.patch)

ifeq ($(OSARCH),Linux64)
PATCHES += $(wildcard numpy-$(VER)-linux64-syslibs.patch)
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install:;
	$(RSYNC) $(PYSITEDIR)/numpy-$(VERSION)-py$(PYTHON_VERSION).egg-info \
		$(PYSITEDIR)/numpy $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
