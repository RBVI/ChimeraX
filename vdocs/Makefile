TOP = ..
include $(TOP)/mk/config.make

SCRIPT = _vdoc.py
VERBOSE = -v

all install build:	check
	$(APP_EXE) --nogui --silent --exit --script '$(SCRIPT) build'
	$(MAKE) -C $(TOP)/docs/user/tools DOCDIR=$$(pwd)/user/tools RSYNC="ln -s" install-icons

check:
	$(PYTHON_EXE) $(SCRIPT) check $(VERBOSE)

clean:	check
	$(PYTHON_EXE) $(SCRIPT) clean

distclean:
	$(PYTHON_EXE) $(SCRIPT) clean
