FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.4.1
DISTRIBUTION = pycollada-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pycollada-$(VERSION)
PATCHES	= $(wildcard pycollada-$(VERSION)-*.patch)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build $(SETUP_BUILD) install

app-install:
	#$(RSYNC) $(PYSITEDIR)/pycollada-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
