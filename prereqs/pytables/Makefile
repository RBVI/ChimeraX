PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.3.0
DISTRIBUTION = tables-$(VERSION).tar.gz
SOURCE = $(tmpdir)/tables-$(VERSION)
HDF_CONFIG = --hdf5='$(build_prefix)'
APP_HDF_CONFIG = --hdf5='$(app_prefix)'

ifdef WIN32
include ../pips/Makefile.pip
WHEEL = tables-$(VERSION)-cp36-cp36m-win_amd64.whl
endif

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install: $(SOURCE)
ifdef WIN32
	$(PIP) install -U $(WHEEL)
else
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py \
		$(HDF_CONFIG) build $(SETUP_BUILD) install
endif

app-install: install
ifdef WIN32
	$(APP_PIP) install $(WHEEL)
else
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/tables-$(VERSION)-py$(PYTHON_VERSION).egg-info \
	#	$(PYSITEDIR)/tables $(APP_PYSITEDIR)
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py \
		$(APP_HDF_CONFIG) build $(SETUP_BUILD) install
endif

$(SOURCE):
ifndef WIN32
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
endif

clean:
	rm -rf $(SOURCE)
