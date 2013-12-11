FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.2.1
DISTRIBUTION = Pillow-$(VERSION).zip
SOURCE = $(tmpdir)/Pillow-$(VERSION)

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

clean:
	rm -rf $(SOURCE)
