FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 06d85a8deb26
DISTRIBUTION = birkenfeld-sphinx-contrib-$(VERSION).zip
SOURCE = $(tmpdir)/birkenfeld-sphinx-contrib-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE)/blockdiag && $(PYTHON_EXE) setup.py $(SETUP_BUILD) install
	cd $(SOURCE)/traclinks && $(PYTHON_EXE) setup.py $(SETUP_BUILD) install

$(SOURCE):
	unzip $(DISTRIBUTION) -d $(tmpdir)

clean:
	rm -rf $(SOURCE)
