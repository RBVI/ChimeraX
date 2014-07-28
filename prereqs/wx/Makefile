FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.0.2.dev76938
DISTRIBUTION = wxPython_Phoenix-$(VERSION).tar.gz
SOURCE = $(tmpdir)/wxPython_Phoenix-$(VERSION)
BUILD_OPTS =
INSTALL_OPTS =

PATCHES = $(wildcard wxPython_Phoenix-$(VERSION)_$(OS)_*.patch) \
	$(wildcard wxPython_Phoenix-$(VERSION)-*.patch)
WAF = $(wildcard waf-1.7.15)

ifdef UNIX
QUOTE_CC := $(subst ','\'',$(CC) $(TARGET_ARCH))
QUOTE_CXX := $(subst ','\'',$(CXX) $(TARGET_ARCH))

# use our compiler options
ENV_CONFIGURE = env CC='$(QUOTE_CC)' CXX='$(QUOTE_CXX)'
CONF_ARGS = --prefix=$(build_prefix) --enable-utf --enable-unicode-properties --disable-static

ifdef DEBUG
MAKE_ARGS = CFLAGS="$(OPT)" CXXFLAGS="$(OPT)"
ifeq ($(OS),Linux)
CONF_ARGS += --enable-valgrind
endif
endif

endif # UNIX

all:

install:	$(SOURCE)
ifeq ($(OS),Windows)
	echo wxPython on Windows not supported yet
else
	rm -f $(libdir)/python$(PYTHON_VERSION)/site-packages/wx/libwx_*
	cd $(SOURCE)/Phoenix && \
		python3 build.py build $(BUILD_OPTS) && \
		python3 build.py install $(INSTALL_OPTS)
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for f in $(PATCHES) ; do \
		( cd $(SOURCE) && patch -p1 ) < $$f ; \
	done
endif
ifneq (,$(WAF))
	cp $(WAF) $(SOURCE)/Phoenix/bin
endif
ifeq ($(OS),Darwin)
	# On Darwin, we patch to add support for selecting
	# OpenGL profile, so we need to rerun sip to get our
	# new constants into Python
	cd $(SOURCE)/Phoenix && \
		python3 build.py build dox && \
		python3 build.py touch && \
		python3 build.py etg --nodoc && \
		python3 build.py sip
endif

clean:
	rm -rf $(SOURCE)
