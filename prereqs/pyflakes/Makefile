FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PYFLAKES_VERSION = main
PYFLAKES_PACKAGE = pyflakes-$(PYFLAKES_VERSION).tar.gz
PYFLAKES_SOURCE = $(tmpdir)/pyflakes-$(PYFLAKES_VERSION)

SETUP_BUILD =
ifdef DEBUG
ifndef WIN32
SETUP_BUILD += --debug
endif
endif

all:


install: $(PYFLAKES_SOURCE)
	cd $(PYFLAKES_SOURCE) && \
		python setup.py $(SETUP_BUILD) install
ifdef USE_MAC_FRAMEWORKS
	cp -p $(PYTHON_FRAMEWORK)/bin/pyflakes $(bindir)
endif

$(PYFLAKES_SOURCE):
	tar zxf $(PYFLAKES_PACKAGE) -C $(tmpdir)

clean:
	rm -rf $(PYFLAKES_SOURCE)
