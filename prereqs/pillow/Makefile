FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.3.1
DISTRIBUTION = Pillow-$(VERSION).zip
SOURCE = $(tmpdir)/Pillow-$(VERSION)

PATCHES	= $(wildcard Pillow-$(VERSION)-*.patch) \
	  $(wildcard Pillow-$(VERSION)_$(OS).patch)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && python3 setup.py $(SETUP_BUILD) install

$(SOURCE):
	unzip $(DISTRIBUTION) -d $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
