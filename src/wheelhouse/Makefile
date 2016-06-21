TOP = ../..
include $(TOP)/mk/config.make

PYSRCS = update_metadata.py cxpi.py

ifdef USE_MAC_FRAMEWORKS
PYPI_SERVER = ../Library/Frameworks/Python.framework/Versions/Current/bin/pypi-server
else
PYPI_SERVER = ../bin/pypi-server
endif

ifdef WIN32
python_exe = ../bin/python
else
python_exe = ../bin/python3
endif

all: $(PYOBJS)

install: dir Makefile.wheelhouse
	$(RSYNC) $(PYSRCS) $(wheelhouse)
	$(RSYNC) Makefile.wheelhouse $(wheelhouse)/Makefile

dir:
	if [ ! -d "$(wheelhouse)" ]; then mkdir -p $(wheelhouse); fi

Makefile.wheelhouse: Makefile.wheelhouse.in
	sed -e 's,PYPI_SERVER,$(PYPI_SERVER),' \
		-e 's,PYTHON_EXE,$(python_exe),' \
		< Makefile.wheelhouse.in > Makefile.wheelhouse

lint:
	$(PYLINT) $(PYSRCS)

clean:
	rm -rf __pycache__ Makefile.wheelhouse
