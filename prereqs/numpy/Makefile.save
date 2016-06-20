PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

ifdef WIN32
VERSION = 1.11.0
else
VERSION = 1.10.1
endif
DISTRIBUTION = numpy-$(VERSION).tar.gz
SOURCE = $(tmpdir)/numpy-$(VERSION)

ifdef WIN32
include ../pips/Makefile.pip
WHEEL = numpy-$(VERSION)+mkl-cp35-cp35m-win_amd64.whl
endif

SETUP_BUILD = build --fcompiler=none
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

VER	:= $(shell expr $(VERSION) : '\(.*\..*\..*\)\.')
ifeq ($(VER),)
VER	= $(VERSION)
endif

PATCHES = $(wildcard numpy-$(VER)-fcompiler.patch)

ifeq ($(OSARCH),Linux64)
PATCHES += $(wildcard numpy-$(VER)-linux64-syslibs.patch)
endif

all:

install: $(SOURCE)
ifdef WIN32
	$(PIP) install -U $(WHEEL)
else
	cd $(SOURCE) && $(PYTHON_EXE) setup.py build $(SETUP_BUILD) install
endif

app-install: install
ifdef WIN32
	$(APP_PIP) install $(WHEEL)
else
	# TODO: figure out way to make below work, so we don't depend on SOURCE
	#$(RSYNC) $(PYSITEDIR)/numpy-$(VERSION)-py$(PYTHON_VERSION).egg-info \
	#	$(PYSITEDIR)/numpy $(APP_PYSITEDIR)
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py build $(SETUP_BUILD) install
endif

$(SOURCE):
ifndef WIN32
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq ($(PATCHES),)
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif
endif

clean:
	rm -rf $(SOURCE) pip.log
