PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

# Build tables from source on Mac and Linux because:
# pypi is missing binary version for macOS,
# and Linux binary links against libhdf5_serial.so.100
# which is only on recent versions of Ubuntu

VERSION = 3.4.4
DISTRIBUTION = tables-$(VERSION).tar.gz
SOURCE = $(tmpdir)/tables-$(VERSION)

ifndef WIN32
FROM_SOURCE = 1
endif

ifdef FROM_SOURCE
SETUP_ARGS= --hdf5=$(build_prefix)
endif

all:

ifndef FROM_SOURCE
install:
	$(PIP_INSTALL) tables==$(VERSION)
else
install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py $(SETUP_ARGS) build install
endif

app-install: install
ifndef FROM_SOURCE
	$(APP_PIP_INSTALL) -U tables==$(VERSION)
else
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) build install
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
ifndef FROM_SOURCE
	rm -rf pip.log
else
	rm -rf $(SOURCE)
endif
