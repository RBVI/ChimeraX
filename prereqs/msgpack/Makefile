PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.5
DISTRIBUTION = msgpack-python-$(VERSION).tar.gz
SOURCE = $(tmpdir)/msgpack-python-$(VERSION)
WHEEL = $(SOURCE)/dist/msgpack_python-$(VERSION)-cp36-cp36m-$(BIN_PLATFORM).whl
ifdef WIN32
WHEEL := $(shell cygpath -m "$(WHEEL)")
endif

all:

install: $(WHEEL)
	$(PYTHON_EXE) -m pip install -I $(WHEEL)

app-install: $(WHEEL)
	$(APP_BOOTSTRAP_PIP) install -I $(WHEEL)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

wheel $(WHEEL): $(SOURCE)
	cd $(SOURCE) &&	$(PYTHON_EXE) setup.py build bdist_wheel

clean:
	rm -rf $(SOURCE)
