PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PYTABLES_VER = 3.6.1

# Binary wheels not available for Mac and Windows for PyTables 3.6.1.
# So use versions we built with Makefile.macos and Makefile.win
ifeq ($(OS),Windows)
DISTRIBUTION = tables-$(PYTABLES_VER)-cp39-cp39-win_amd64.whl
endif
ifeq ($(OS),Darwin)
DISTRIBUTION = tables-$(PYTABLES_VER)-py$(PYTHON_VERSION)-mac.tar.gz
endif

ifeq ($(OS),Linux)
app-install:
	$(APP_PYTHON_EXE) -m pip install tables==$(PYTABLES_VER)
endif

ifeq ($(OS),Windows)
app-install:
	$(APP_PYTHON_EXE) -m pip install $(DISTRIBUTION)
endif

ifeq ($(OS),Darwin)
app-install:
	tar xf $(DISTRIBUTION) -C $(APP_PYSITEDIR)
	$(APP_PYTHON_EXE) -m pip install numexpr
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/pytables/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/pytables

make_mac_distribution:
	make -f Makefile.macos

clean:
	rm -rf $(DISTRIBUTION)
