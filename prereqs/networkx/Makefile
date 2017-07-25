PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2.0-july-24-2017
DISTRIBUTION = networkx-$(VERSION).tar.gz
SOURCE = $(tmpdir)/networkx

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build install

app-install: install
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py build install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
