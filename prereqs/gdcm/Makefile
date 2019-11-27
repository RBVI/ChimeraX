PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Use PIP_INSTALL
include ../pips/Makefile.pip

# Using OpenMM builds from https://anaconda.org/omnia/openmm/files
PYVER = py37
VERSION = 2.8.9
ifeq ($(OS),Darwin)
PLATFORM = mac
endif
ifeq ($(OS),Linux)
PLATFORM = linux
endif
ifeq ($(OS),Windows)
PLATFORM = windows
WHEEL = gdcm-2.8.8-cp37-cp37m-win_amd64.whl
endif

DISTRIBUTION = gdcm-$(VERSION)-$(PLATFORM)-$(PYVER).tar.bz2
SOURCE = $(tmpdir)/gdcm-$(VERSION)
GDCM_SITEDIR = gdcm-$(VERSION)/lib/python3.7/site-packages

ifeq ($(OS),Windows)

install:
	$(PIP_INSTALL) install $(WHEEL)

app-install:
	$(APP_PIP_INSTALL) install $(WHEEL)

clean:

else

install: $(SOURCE)
	$(RSYNC) $(SOURCE) $(PYSITEDIR)
	ln -sf $(GDCM_SITEDIR)/gdcm.py $(GDCM_SITEDIR)/gdcmswig.py $(GDCM_SITEDIR)/_gdcmswig.so $(PYSITEDIR)

app-install: $(SOURCE)
	$(RSYNC) $(SOURCE) $(APP_PYSITEDIR)
	ln -sf $(GDCM_SITEDIR)/gdcm.py $(GDCM_SITEDIR)/gdcmswig.py $(GDCM_SITEDIR)/_gdcmswig.so $(APP_PYSITEDIR)

$(SOURCE):
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)

clean:
	rm -rf $(SOURCE)

endif
