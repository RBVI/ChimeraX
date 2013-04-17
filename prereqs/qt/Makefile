FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

QT_VERSION = 4.8.4
QT_PACKAGE = qt-everywhere-opensource-src-$(QT_VERSION).tar.gz
QT_SOURCE = $(tmpdir)/qt-everywhere-opensource-src-$(QT_VERSION)

PATCHES	= $(wildcard qt-src-$(QT_VERSION)-*.patch)

ifdef USE_MAC_FRAMEWORK
ENV = env CC='gcc -I$(includedir) -F$(frameworkdir)' \
	CXX='g++ -I$(includedir) -F$(frameworkdir)'
endif

CONFIG_OPTS = -opensource -confirm-license \
	      -no-qt3support -no-webkit \
	      -no-multimedia -no-audio-backend -no-phonon \
	      -qt-libtiff -qt-libpng -qt-libmng -qt-libjpeg
ifeq ($(OS),Windows)
	CONFIG_OPTS += -mp -prefix '$(shell cygpath -m '$(build_prefix)')'
else
	CONFIG_OPTS += -prefix '$(build_prefix)' -prefix-install -no-glib
endif
ifdef MAC_USE_FRAMEWORKS
	CONFIG_OPTS += -framework
endif
ifeq ($(OS),Darwin)
	CONFIG_OPTS += -sdk $(SDK)
endif
ifdef DEBUG
	CONFIG_OPTS += -debug
else
	CONFIG_OPTS += -release
endif

all:

install: $(QT_SOURCE)
ifeq ($(OS),Windows)
	cd $(QT_SOURCE) && ./configure.exe $(CONFIG_OPTS) && nmake && nmake install
else
	cd $(QT_SOURCE) && $(ENV) ./configure $(CONFIG_OPTS) && $(MAKE) && $(MAKE) -j1 install
endif
	rm -f $(build_prefix)/q3porting.xml
ifdef USE_MAC_FRAMEWORKS
	cd $(libdir) && for i in *.framework; do \
		f=`basename $$i .framework`; \
		(cd ../include && ln -fs ../lib/$$i/Headers $$f); \
	done
endif

$(QT_SOURCE): $(QT_PACKAGE)
	tar zxf $(QT_PACKAGE) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(QT_SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(QT_SOURCE)
