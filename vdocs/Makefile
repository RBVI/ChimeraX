TOP = ..
include $(TOP)/mk/config.make

SCRIPT = _vdoc.py
VERBOSE = -v

all install build:	check
	rm -f user/index.html
	$(APP_EXE) --nogui --silent --exit --script '$(SCRIPT) build'
	$(MAKE) -C $(TOP)/docs/user/tools DOCDIR=$$(pwd)/user/tools RSYNC="ln -fs" install-icons

ifneq (,$(wildcard $(PYTHON_EXE)))
check:
	$(PYTHON_EXE) $(SCRIPT) check $(VERBOSE)

clean:	check
	$(PYTHON_EXE) $(SCRIPT) clean

distclean:
	$(PYTHON_EXE) $(SCRIPT) clean
else
clean distclean:
	rm -rf user
endif

from-daily:
	rm -rf user
	bin=$(shell dirname $$(readlink $$(which chimerax-daily))); \
	$(MAKE) APP_EXE=chimerax-daily PYTHON_EXE="$$bin/python$(PYTHON_VERSION)" install
