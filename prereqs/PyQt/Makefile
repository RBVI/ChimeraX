PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.6
DISTRIBUTION = PyQt5_gpl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt5_gpl-$(VERSION)
# versions prior to 5.6...
#DISTRIBUTION = PyQt-gpl-$(VERSION).tar.gz
#SOURCE = $(tmpdir)/PyQt-gpl-$(VERSION)

ifeq ($(OS),Darwin)
SIP_BIN = $(PYTHON_FRAMEWORK)/bin
else
SIP_BIN = $(bindir)
endif
CONFIG_ARGS = --confirm-license --verbose --sip $(SIP_BIN)/sip --qmake $(bindir)/qmake

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(frameworkdir)", "$(bindir)"))')

all:

install: $(SOURCE)
	cd $(SOURCE) && $(PYTHON_EXE) configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install

app-install:
	$(RSYNC) $(PYSITEDIR)/PyQt5 $(APP_PYSITEDIR)

source: $(SOURCE)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
