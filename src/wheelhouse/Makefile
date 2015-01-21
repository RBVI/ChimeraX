TOP = ../..
include $(TOP)/mk/config.make

PYSRCS = update_metadata.py

ifdef USE_MAC_FRAMEWORKS
PYPI_SERVER = ../Frameworks/Python.framework/Versions/Current/bin/pypi-server
else
PYPI_SERVER = ../bin/pypi-server
endif

all: $(PYOBJS)

install: dir Makefile.wheelhouse
	$(RSYNC) $(PYSRCS) $(wheelhouse)
	$(RSYNC) Makefile.wheelhouse $(wheelhouse)/Makefile

dir:
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

Makefile.wheelhouse: Makefile.wheelhouse.in
	sed -e 's,PYPI_SERVER,$(PYPI_SERVER),' \
		< Makefile.wheelhouse.in > Makefile.wheelhouse

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ Makefile.wheelhouse
