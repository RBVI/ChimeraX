PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

#
# The imagecodecs-lite 2019.2.22 package requires Cython on Mac and Linux
# (has binary wheels on Windows) to compile from source code, but it does
# not specify the cython dependency.
# Also its setup.py uses Cython before dependencies are installed during
# "python setup.py egg_info" which is run by pip install which gives an error.
# Because of these # bugs in the imagecodecs-lite PyPi scripts,
# it has to be installed after Cython is installed by prereqs/pips.
# I wrote to the imagecodecs-lite author Chris Gohlke about these problems.
# If they get fixed then imagecodecs-lite can be installed in prereqs/pips
# like all the other PyPi packages used by ChimeraX.
#

VERSION = 2019.4.20

all:

install:
	$(PIP_INSTALL) imagecodecs-lite==$(VERSION)

app-install: install
	$(APP_PIP_INSTALL) -U imagecodecs-lite==$(VERSION)

clean:
	rm -rf pip.log
