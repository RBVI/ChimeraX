FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 15.1

# Need to install from source, so don't put in pips

all:

install:
	$(PYTHON_EXE) -m pip install --no-use-wheel setuptools==$(VERSION)

app-install:
	$(APP_PYTHON_EXE) -m pip install --no-use-wheel setuptools==$(VERSION)

clean:
