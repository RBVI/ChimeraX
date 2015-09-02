#
# source fetched from http://wxpython.org/Phoenix/snapshot-builds/
#

FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.0.3.dev1719+62a2221
SHLIB_VER = 3.0.0.3.0
DISTRIBUTION = wxPython_Phoenix-$(VERSION).tar.gz
SOURCE = $(tmpdir)/wxPython_Phoenix-$(VERSION)
BUILD_OPTS =
INSTALL_OPTS =
ifdef DEBUG
BUILD_OPTS += --debug
INSTALL_OPTS += --debug
endif

ifeq ($(OS),Linux)
SYSLIBDIR = /usr/lib/x86_64-linux-gnu
SYSLIBS = libjpeg.so.62 libwebkitgtk-1.0.so.0 libjavascriptcoregtk-1.0.so.0
endif

PATCHES = $(wildcard wxPython_Phoenix-$(VERSION)_$(OS)_*.patch) \
	$(wildcard wxPython_Phoenix-$(VERSION)-*.patch)
PATCHINES = $(wildcard wxPython_Phoenix-$(VERSION)_$(OS)_*.patch.in) \
	$(wildcard wxPython_Phoenix-$(VERSION)-*.patch.in)
WAF = $(wildcard waf-1.7.15)

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(PYSITEDIR)/wx", "$(bindir)"))')

all:

install:	$(SOURCE)
ifeq ($(OS),Windows)
	echo "wxPython on Windows not supported yet" && exit 1
else
	rm -f $(PYSITEDIR)/wx/libwx_*
	cd $(SOURCE) && \
		$(PYTHON_EXE) build.py build $(BUILD_OPTS) && \
		$(PYTHON_EXE) build.py install $(INSTALL_OPTS)
ifeq ($(OS),Darwin)
	cd $(PYSITEDIR)/wx; for f in *-$(SHLIB_VER).dylib; do \
		install_name_tool -id \
			@executable_path/$(RELPATH)/$$f $$f; \
	done
endif
ifeq ($(OS),Linux)
	cd $(PYSITEDIR)/wx; mv -f libwx_* $(libdir)/
	cd $(SYSLIBDIR) ; $(RSYNC) --copy-links $(SYSLIBS) $(shlibdir)
endif
endif

app-install:
	$(RSYNC) $(PYSITEDIR)/wxPython_Phoenix-$(VERSION)-py$(PYTHON_VERSION).egg-info \
		$(PYSITEDIR)/wx $(APP_PYSITEDIR)
ifndef USE_MAC_FRAMEWORKS
ifneq ($(OS),Darwin)
	$(RSYNC) $(shlibdir)/libwx_*$(SHLIB_EXT)* $(app_shlibdir)
	cd $(shlibdir) ; $(RSYNC) $(SYSLIBS) $(app_shlibdir)
endif
endif

$(SOURCE) source:
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
	cp $(WAF) $(SOURCE)/bin
endif
ifneq (,$(PATCHES)$(PATCHINES))
	# Since we patched wx, we need to rerun sip to incorporate our changes
	cd $(SOURCE) && \
		$(PYTHON_EXE) build.py build dox && \
		$(PYTHON_EXE) build.py touch && \
		$(PYTHON_EXE) build.py etg --nodoc && \
		$(PYTHON_EXE) build.py sip
endif

clean:
	rm -rf $(SOURCE)
