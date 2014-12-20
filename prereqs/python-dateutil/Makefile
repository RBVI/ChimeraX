FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.3
DISTRIBUTION = python-dateutil-$(VERSION).tar.gz
SOURCE = $(tmpdir)/python-dateutil-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install

app-install:
	$(RSYNC) $(PYSITEDIR)/python_dateutil-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)


$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
