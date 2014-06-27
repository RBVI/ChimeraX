FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.6
DISTRIBUTION = Pygments-$(VERSION).tar.gz
SOURCE = $(tmpdir)/Pygments-$(VERSION)

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
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
