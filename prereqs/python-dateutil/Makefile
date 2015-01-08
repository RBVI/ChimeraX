FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.3
DISTRIBUTION = python-dateutil-$(VERSION).tar.gz
SOURCE = $(tmpdir)/python-dateutil-$(VERSION)
DATEUTIL_EGG = python_dateutil-$(VERSION)-py$(PYTHON_VERSION).egg

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install
ifeq ($(OS),Darwin)
# Unzip egg so dateutil source code is available for Mac app install.
	cd $(PYSITEDIR) ; unzip $(DATEUTIL_EGG) -x "EGG-INFO/*"
endif

app-install:
	$(RSYNC) $(PYSITEDIR)/$(DATEUTIL_EGG) $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
