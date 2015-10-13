FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.2.2
DISTRIBUTION = tables-$(VERSION).tar.gz
SOURCE = $(tmpdir)/tables-$(VERSION)
HDF_CONFIG = --hdf5='$(build_prefix)'
APP_HDF_CONFIG = --hdf5='$(app_prefix)'

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py \
		$(HDF_CONFIG) build $(SETUP_BUILD) install

app-install: install
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/tables-$(VERSION)-py$(PYTHON_VERSION).egg-info \
	#	$(PYSITEDIR)/tables $(APP_PYSITEDIR)
	cd $(SOURCE) &&	$(APP_PYTHON_EXE) setup.py \
		$(APP_HDF_CONFIG) build $(SETUP_BUILD) install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
