PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.3.1
EGG_VERSION = 1.3
DISTRIBUTION = pdbfixer-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/pdbfixer-$(VERSION)

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install

app-install: install
	$(RSYNC) $(PYSITEDIR)/pdbfixer-$(EGG_VERSION)-py$(PYTHON_VERSION).egg $(APP_PYSITEDIR)

$(SOURCE):
	tar xf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
