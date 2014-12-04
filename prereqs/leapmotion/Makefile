FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Leap library is built for Python 3.4.
VERSION = 1.2.0
DISTRIBUTION = leap-$(VERSION).tar.gz
SOURCE = $(tmpdir)/leap-$(VERSION)

all:

install: $(SOURCE)
ifeq ($(OS),Darwin)
	-mkdir $(PYSITEDIR)/LeapMotion
	$(RSYNC) $(SOURCE)/darwin/ $(PYSITEDIR)/LeapMotion
# Fix the path to the Python library
	install_name_tool -change $(build_prefix)/Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/Python $(PY_LIB) @executable_path/../Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/Python $(PYSITEDIR)/LeapMotion/LeapPython.so
endif

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
