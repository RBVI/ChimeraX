FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.0.2.dev78088
SHLIB_VER = 3.0.0.3.0
DISTRIBUTION = wxPython_Phoenix-$(VERSION).tar.gz
SOURCE = $(tmpdir)/wxPython_Phoenix-$(VERSION)
BUILD_OPTS =
INSTALL_OPTS =

PATCHES = $(wildcard wxPython_Phoenix-$(VERSION)_$(OS)_*.patch) \
	$(wildcard wxPython_Phoenix-$(VERSION)-*.patch)
PATCHINES = $(wildcard wxPython_Phoenix-$(VERSION)_$(OS)_*.patch.in) \
	$(wildcard wxPython_Phoenix-$(VERSION)-*.patch.in)
WAF = $(wildcard waf-1.7.15)

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(PYSITEDIR)", "$(bindir)"))')

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
	rm -f $(PYSITEDIR)/wx/libwx_*
	cd $(SOURCE)/Phoenix && \
		$(PYTHON_EXE) build.py build $(BUILD_OPTS) && \
		$(PYTHON_EXE) build.py install $(INSTALL_OPTS)
ifeq ($(OS),Darwin)
	cd $(PYSITEDIR)/wx; for f in *-$(SHLIB_VER).dylib; do \
		install_name_tool -id \
			@executable_path/$(RELPATH)/$$f $$f; \
	done
endif
endif

app-install:
	$(RSYNC) $(PYSITEDIR)/wxPython_Phoenix-$(VERSION)-py$(PYTHON_VERSION).egg-info \
		$(PYSITEDIR)/wx $(APP_PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for f in $(PATCHES) ; do \
		( cd $(SOURCE) && patch -p1 ) < $$f ; \
	done
endif
ifneq (,$(PATCHINES))
	for f in $(PATCHINES) ; do \
		sed 's,SDK_PATH,"$(SYSROOT)",' $$f \
			| ( cd $(SOURCE) && patch -p1 ) ; \
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
		$(PYTHON_EXE) build.py build dox && \
		$(PYTHON_EXE) build.py touch && \
		$(PYTHON_EXE) build.py etg --nodoc && \
		$(PYTHON_EXE) build.py sip
endif

clean:
	rm -rf $(SOURCE)
