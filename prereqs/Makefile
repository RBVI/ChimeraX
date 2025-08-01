# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

# if CDPATH is set, then scripts that invoke cd will echo the new directory
# and potentially screw up constructed files.
unexport CDPATH

TOP	= ..
NO_SUBDIR_TEST = 1

include $(TOP)/mk/config.make

all:
	@echo "'make install' to build everything" \
	&& echo "or 'make SUBIDR.install' to install that"

#
# Set your various NEEDs above.
#

ifndef UBUNTU_DIST
ifndef FLATPAK_DIST
NEED_FFMPEG=1
endif
endif

ifdef WIN32
SUBDIRS += win32
endif

ifdef NEED_FFMPEG
SUBDIRS += ffmpeg
else
ffmpeg.install:
ffmpeg.app-install:
endif


# Python egg installation has a race condition updating easy-install.pth
.NOTPARALLEL:

# keep in alphabetical order, dependencies go below

SUBDIRS	+= ambertools pips
SUBDIRS += Python

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

$(SUBDIR_INSTALL): build-dirs

build-dirs:
	$(MAKE) -C $(TOP) build-dirs

build-app-dirs:
	$(MAKE) -C $(TOP) build-app-dirs

ifdef NO_PREBUILT
prebuilt:

install-prebuilt:

else
ifeq ($(OS),Linux)
GCC_VER = $(shell $(CC) -dumpversion)
PREBUILT = prebuilt-$(OSARCH)-$(GCC_VER).tar.bz2
else ifeq ($(OS),Darwin)
PREBUILT = prebuilt-$(OSARCH).tar.bz2
else ifeq ($(OS),Windows)
PREBUILT = prebuilt-$(OSARCH).tar.bz2
else
PREBUILT = prebuilt-$(shell uname -s)-$(shell uname -m).tar.bz2
endif

prebuilt:
	prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar --exclude ./tmp -cjf $$prebuilt .

install-prebuilt:
	@echo 'Started prereq install at' `date` on `hostname`
	if [ -e $(PREBUILT) ]; then \
		$(MAKE) build-dirs ; \
		prebuilt=`pwd`/$(PREBUILT) && cd $(build_prefix) && tar xjf $$prebuilt ; \
	else \
		$(MAKE) install && $(MAKE) prebuilt; \
	fi
	@echo 'Finished prereq install at' `date`
endif

SUBDIR_APP_INSTALL = $(SUBDIRS:=.app-install)
.PHONY: $(SUBDIR_APP_INSTALL)
app-install: $(SUBDIR_APP_INSTALL)

$(SUBDIR_APP_INSTALL):
	$(MAKE) -C $(subst .app-install,,$@) app-install
$(SUBDIR_APP_INSTALL): build-app-dirs

# Whole platform dependencies
ifdef WIN32
$(subst win32.install,,$(SUBDIR_INSTALL)): win32.install
endif

# Please keep dependencies in alphabetical order.
# All packages that have dependencies should depend on the .install
# target of the other packages.

pips.app-install: Python.app-install
pips.install: Python.install
