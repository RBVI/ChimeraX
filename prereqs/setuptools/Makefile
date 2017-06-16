PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 34.4.1

# Need to install from source, so don't put in pips

all:

install:
	$(PYTHON_EXE) -m pip install setuptools==$(VERSION)

app-install:
	$(APP_PYTHON_EXE) -m pip install setuptools==$(VERSION)

clean:
