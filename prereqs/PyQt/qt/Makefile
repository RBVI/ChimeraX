# Qt dependencies given in:
#
#	http://qt-project.org/wiki/Building_Qt_5_from_Git

PREREQ_MAKE = 1
TOP	= ../../..
include $(TOP)/mk/config.make

VERSION = 5.9
MINOR_VERSION = 1
FULL_VERSION = $(VERSION).$(MINOR_VERSION)
DISTRIBUTION = qt-everywhere-opensource-src-$(FULL_VERSION).tar.xz
SOURCE = $(tmpdir)/qt-everywhere-opensource-src-$(FULL_VERSION)

PATCHES	= $(wildcard qt-$(FULL_VERSION)-*.patch) $(wildcard qt-$(FULL_VERSION)_$(OS)-*.patch)

CONFIG_OPTS = -opensource -confirm-license \
	      -qt-libpng -qt-libjpeg \
	      -nomake examples
ifeq ($(OS),Windows)
	CONFIG_OPTS += -mp -prefix '$(shell cygpath -m '$(build_prefix)')'
else
	CONFIG_OPTS += -prefix '$(build_prefix)' -no-glib
endif
ifdef MAC_USE_FRAMEWORKS
	CONFIG_OPTS += -framework
endif
#ifeq ($(OS),Darwin)
#	CONFIG_OPTS += -sdk $(SDK)
#endif
ifeq ($(OS),Linux)
	# In 5.0.2, build fails if icu library is not present,
	# so have configure fail if it is not present
	CONFIG_OPTS += -qt-xcb -icu
endif
ifdef DEBUG
	CONFIG_OPTS += -debug
else
	CONFIG_OPTS += -release
endif

RELPATH = $(shell $(PYTHON_EXE) -c 'import os; print(os.path.relpath("$(frameworkdir)", "$(bindir)"))')

all:

install: $(SOURCE)
ifeq ($(OS),Windows)
	cd $(SOURCE) && ./configure.exe $(CONFIG_OPTS) && nmake && nmake install
else
	# To get the below to work with Xcode 8, I had to:
	# cd /Applications/Xcode.app/Contents/Developer/usr/bin/
	# sudo ln -s xcodebuild xcrun
	cd $(SOURCE) && $(ENV) ./configure $(CONFIG_OPTS) && $(MAKE) && $(MAKE) -j1 install
endif
	rm -f $(build_prefix)/q3porting.xml
ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(libdir)/Qt*.framework $(frameworkdir)
	$(MAKE) include

include:
	cd $(libdir) && for i in *.framework; do \
		f=`basename $$i .framework`; \
		(cd ../include && ln -nfs ../lib/$$i/Headers $$f); \
	done
endif

app-install:
ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(frameworkdir)/Qt*.framework $(app_libdir)
	$(RSYNC) $(build_prefix)/plugins $(app_libdir)/..
endif

$(SOURCE): $(DISTRIBUTION)
	$(bindir)/7za x -o'$(tmpdir)' $(DISTRIBUTION)
	(cd $(tmpdir); tar zxf $(SOURCE).tar)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p1) < $$p ; \
	done
	touch $(SOURCE)
endif

clean:
	rm -rf $(SOURCE)
