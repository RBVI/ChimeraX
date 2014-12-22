FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.8.0
DISTRIBUTION = six-$(VERSION).tar.gz
SOURCE = $(tmpdir)/six-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install

app-install:
	$(RSYNC) $(PYSITEDIR)/six-$(VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
