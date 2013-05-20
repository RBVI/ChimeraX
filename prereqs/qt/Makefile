FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 5.0.2
DISTRIBUTION = qt-everywhere-opensource-src-$(VERSION).tar.gz
SOURCE = $(tmpdir)/qt-everywhere-opensource-src-$(VERSION)

PATCHES	= $(wildcard qt-src-$(VERSION)-*.patch)

ifdef USE_MAC_FRAMEWORK
ENV = env CC='gcc -I$(includedir) -F$(frameworkdir)' \
	CXX='g++ -I$(includedir) -F$(frameworkdir)'
endif

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
	CONFIG_OPTS += -qt-xcb
endif
ifdef DEBUG
	CONFIG_OPTS += -debug
else
	CONFIG_OPTS += -release
endif

all:

install: $(SOURCE)
ifeq ($(OS),Windows)
	cd $(SOURCE) && ./configure.exe $(CONFIG_OPTS) && nmake && nmake install
else
	cd $(SOURCE) && $(ENV) ./configure $(CONFIG_OPTS) && $(MAKE) && $(MAKE) -j1 install
endif
	rm -f $(build_prefix)/q3porting.xml
ifdef USE_MAC_FRAMEWORKS
	cd $(libdir) && for i in *.framework; do \
		f=`basename $$i .framework`; \
		(cd ../include && ln -fs ../lib/$$i/Headers $$f); \
	done
endif

$(SOURCE): $(DISTRIBUTION)
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
