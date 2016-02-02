FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.5.1
DISTRIBUTION = PyQt-gpl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt-gpl-$(VERSION)

ifeq ($(OS),Darwin)
SIP_BIN = $(PYTHON_FRAMEWORK)/bin
else
SIP_BIN = $(bindir)
endif
CONFIG_ARGS = --confirm-license --verbose --sip $(SIP_BIN)/sip --qmake $(bindir)/qmake

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install

app-install:
	$(RSYNC) $(PYSITEDIR)/PyQt5 $(APP_PYSITEDIR)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
