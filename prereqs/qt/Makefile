# Qt dependencies given in:
#
#	http://qt-project.org/wiki/Building_Qt_5_from_Git

FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.5.1
DISTRIBUTION = qt-everywhere-opensource-src-$(VERSION).7z
SOURCE = $(tmpdir)/qt-everywhere-opensource-src-$(VERSION)

PATCHES	= $(wildcard qt-$(VERSION)-*.patch) $(wildcard qt-$(VERSION)_$(OS)-*.patch)

CONFIG_OPTS = -opensource -confirm-license \
	      -no-audio-backend \
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
ifeq ($(OS),Darwin)
	CONFIG_OPTS += -sdk $(SDK)
endif
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
	#cd $(SOURCE) && $(ENV) ./configure $(CONFIG_OPTS) && $(MAKE) && $(MAKE) -j1 install
endif
	rm -f $(build_prefix)/q3porting.xml
ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(libdir)/Qt*.framework $(frameworkdir)
	# fix rpaths in shared libs
	for fw in $(frameworkdir)/Qt*.framework; do \
		fwlib=$$fw/Versions/Current/`basename -s .framework $$fw`; \
		for linked in `otool -L $$fwlib | grep Qt | grep framework | grep rpath | awk '{ print $$1 }'`; do \
			fixed=`echo $$linked | sed 's|@rpath|@executable_path/$(RELPATH)|'`; \
			install_name_tool -change "$$linked" "$$fixed" "$$fwlib"; \
		done; \
	done
	$(MAKE) include

include:
	cd $(libdir) && for i in *.framework; do \
		f=`basename $$i .framework`; \
		(cd ../include && ln -nfs ../lib/$$i/Headers $$f); \
	done
endif

app-install:
ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) $(frameworkdir)/Qt*.framework $(app_frameworkdir)
endif

$(SOURCE): $(DISTRIBUTION)
	$(bindir)/7za x -o'$(tmpdir)' $(DISTRIBUTION)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
