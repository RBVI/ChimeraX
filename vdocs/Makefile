TOP = ..
include $(TOP)/mk/config.make

SCRIPT = _vdoc.py
VERBOSE = -v

all install build:	check
	rm -f user/index.html
	$(APP_EXE) --nogui --silent --exit --script '$(SCRIPT) build'
	$(MAKE) -C $(TOP)/docs/user/tools DOCDIR=$$(pwd)/user/tools RSYNC="ln -fs" install-icons

ifeq (,$(wildcard $(APP_EXE)))
# No application means an incomplete installation

clean distclean:
	rm -rf user

else
check:
	$(APP_EXE) --nogui --safemode --exit --script '$(SCRIPT) check $(VERBOSE)'

clean:	check
	$(APP_EXE) --nogui --safemode --exit --script '$(SCRIPT) clean'

distclean:
	$(APP_EXE) --nogui --safemode --exit --script '$(SCRIPT) clean'
endif

from-daily:
	rm -rf user
	$(MAKE) APP_EXE="$$(which chimerax-daily)" install
