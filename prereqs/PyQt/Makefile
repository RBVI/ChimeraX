FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.2.1
DISTRIBUTION = PyQt-gpl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt-gpl-$(VERSION)

CONFIG_ARGS = --confirm-license --verbose --sip $(PYTHON_FRAMEWORK)/bin/sip --qmake $(bindir)/qmake

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install

app-install:;

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
