FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Leap library is built for Python 3.4.
VERSION = 1.2.0
DISTRIBUTION = leap-$(VERSION).tar.gz
SOURCE = $(tmpdir)/leap-$(VERSION)
LEAP_PYTHON = /Users/goddard/ucsf/chimera2/build/Library/Frameworks/Python.framework/Versions/3.4/Python
INSTALL_DIR = $(PYSITEDIR)/LeapMotion

PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
REL_PY_LIB = $(shell python -c 'import os; print(os.path.relpath("$(PY_LIB)", "$(bindir)"))')

all:

install: $(SOURCE)
ifeq ($(OS),Darwin)
	-mkdir $(PYSITEDIR)/LeapMotion
	$(RSYNC) $(SOURCE)/darwin/ $(INSTALL_DIR)
# Fix the path to the Python library
	install_name_tool -change $(LEAP_PYTHON) @executable_path/$(REL_PY_LIB) $(INSTALL_DIR)/LeapPython.so
endif

app-install:
ifeq ($(OS),Darwin)
	$(RSYNC) $(PYSITEDIR)/LeapMotion $(APP_PYSITEDIR)
else
	;
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
