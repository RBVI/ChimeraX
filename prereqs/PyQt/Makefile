FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.5.1
DISTRIBUTION = PyQt-gpl-$(VERSION).tar.gz
SOURCE = $(tmpdir)/PyQt-gpl-$(VERSION)

ifeq ($(OS),Darwin)
SIP_BIN = $(PYTHON_FRAMEWORK)/bin
else
SIP_BIN = $(bindir)
endif
CONFIG_ARGS = --confirm-license --verbose --sip $(SIP_BIN)/sip --qmake $(bindir)/qmake

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(frameworkdir)", "$(bindir)"))')

all:

install: $(SOURCE)
	#cd $(SOURCE) && $(PYTHON_EXE) configure.py $(CONFIG_ARGS) && $(MAKE) && $(MAKE) install
	$(RSYNC) $(build_prefix)/plugins $(PYSITEDIR)/PyQt5
ifdef USE_MAC_FRAMEWORKS
	# Have the PyQt modules look in the right place for Qt frameworks
	cd $(PYSITEDIR)/PyQt5; for mod in *.$(PYMOD_EXT); do \
		for fw in `otool -L $$mod | grep Qt | grep framework | grep rpath | awk '{ print $$1 }'`; do \
			fixed=`echo $$fw | sed 's|@rpath|@executable_path/$(RELPATH)|'`; \
			install_name_tool -change "$$fw" "$$fixed" "$$mod"; \
		done; \
	done
endif

app-install:
	$(RSYNC) $(PYSITEDIR)/PyQt5 $(APP_PYSITEDIR)

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
