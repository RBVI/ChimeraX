TOP = .
TOP := $(shell (cd "$(TOP)"; pwd))
NO_SUBDIR_ALL=1
NO_SUBDIR_INSTALL=1
SUBDIRS = prereqs src

include $(TOP)/mk/config.make
include $(TOP)/mk/subdir.make

all:
	@echo "'make install' to build everything"

install:
	@echo 'Started install at' `date` on `hostname`
	$(MAKE) build-dirs
	$(MAKE) -C prereqs install-prebuilt
	$(MAKE) -C prereqs app-install
	$(MAKE) -C src install
	$(MAKE) -C docs html
	@echo 'Finished install at' `date`

docs.install:
	$(MAKE) -C docs html

build-dirs:
	-mkdir $(build_prefix) $(bindir) $(libdir) $(includedir) $(datadir) \
		$(webdir) $(wheelhouse)
	-cd $(build_prefix) && ln -nfs lib lib64
ifneq ($(libdir), $(shlibdir))
	-mkdir $(shlibdir)
endif
ifeq ($(OS),Darwin)
	-mkdir -p $(frameworkdir) $(build_prefix)/Library
	#-cd $(build_prefix)/Library && ln -nfs ../Frameworks .
endif

build-app-dirs:
	-mkdir -p $(app_prefix) $(app_bindir) $(app_libdir) $(app_datadir) \
		$(app_includedir) $(APP_PYSITEDIR)
ifeq ($(OS),Darwin)
	-mkdir -p $(app_prefix)/MacOS $(app_prefix)/Resources \
		$(app_frameworkdir)
endif

distclean: clean
	rm -rf $(build_prefix) $(app_prefix) prereqs/prebuilt-*.tar.bz2

build-from-scratch:
	$(MAKE) distclean
	$(MAKE) install

# Linux debugging:

gltrace:
	rm -f $(APP_NAME).trace
	apitrace trace $(app_bindir)/$(APP_NAME) $(DATA_FILE)

dumptrace:
	@apitrace dump $(APP_NAME).trace
