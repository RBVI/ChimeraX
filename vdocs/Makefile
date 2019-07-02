TOP = ..
include $(TOP)/mk/config.make

SCRIPT = _vdoc.py
VERBOSE = -v

all install build:	check
	$(PYTHON_EXE) $(SCRIPT) build

check:
	$(PYTHON_EXE) $(SCRIPT) check $(VERBOSE)

clean:	check
	$(PYTHON_EXE) $(SCRIPT) clean

distclean:
	$(PYTHON_EXE) $(SCRIPT) clean
