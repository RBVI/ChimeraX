PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.3.1
DISTRIBUTION = pdbfixer-$(VERSION).tar.bz2
SOURCE = $(tmpdir)/pdbfixer-$(VERSION)

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py install

app-install: install
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py install

$(SOURCE):
	tar xf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
