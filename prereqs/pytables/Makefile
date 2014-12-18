FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.1.1
DISTRIBUTION = tables-$(VERSION).tar.gz
SOURCE = $(tmpdir)/tables-$(VERSION)
HDF_CONFIG = --hdf5='$(build_prefix)'

all:

install: $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py $(HDF_CONFIG) install

app-install:
	$(RSYNC) $(PYSITEDIR)/tables-$(VERSION)-py$(PYTHON_VERSION).egg-info \
		$(PYSITEDIR)/tables $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
