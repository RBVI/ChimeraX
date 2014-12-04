FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.6.1
DISTRIBUTION = Pillow-$(VERSION).tar.gz
SOURCE = $(tmpdir)/Pillow-$(VERSION)
PILLOW_EGG = $(wildcard Pillow-$(VERSION)-py$(PYTHON_VERSION)-*.egg)
JPEG_INC = $(build_prefix)/include
JPEG_LIB = $(build_prefix)/lib

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
	cd $(SOURCE) && $(ENV) INCLUDE=$(JPEG_INC) LIB=$(JPEG_LIB) $(PYTHON_EXE) setup.py $(SETUP_BUILD) build_ext --disable-tk install
# Unzip egg so PIL source code is available for Mac app install.
ifneq (,$(PILLOW_EGG))
	cd $(PYSITEDIR) ; unzip $(PILLOW_EGG) -x "EGG-INFO/*"
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
