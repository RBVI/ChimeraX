FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.4
DISTRIBUTION = webcolors-$(VERSION).tar.gz
SOURCE = $(tmpdir)/webcolors-$(VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install:;
	$(RSYNC) $(PYSITEDIR)/webcolors-$(VERSION)-py$(PYTHON_VERSION).egg-info \
		$(PYSITEDIR)/webcolors.py $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
