TOP = .
TOP := $(shell (cd "$(TOP)"; pwd))
NO_SUBDIR_ALL=1
NO_SUBDIR_INSTALL=1
NO_SUBDIR_DISTCLEAN=1
SUBDIRS = prereqs src

include $(TOP)/mk/config.make
include $(TOP)/mk/subdir.make

all:
	@echo "'make install' to build everything"

install:
	$(MAKE) build-dirs
	$(MAKE) -C prereqs install.prebuilt
	$(MAKE) -C src install
	$(MAKE) -C docs html

build-dirs:
	-mkdir $(build_prefix) $(bindir) $(libdir) $(includedir) $(datadir)
ifneq ($(libdir), $(shlibdir))
	-mkdir $(shlibdir)
endif
ifdef USE_MAC_FRAMEWORK
	-mkdir -p $(frameworkdir)
endif

distclean: clean
#	rm -rf $(build_prefix)
