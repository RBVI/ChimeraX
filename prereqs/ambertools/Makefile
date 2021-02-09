PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using AmberTools distribution from https://ambermd.org

VERSION = 20

DISTRIBUTION = AmberTools$(VERSION).tar.bz2
SOURCE = $(tmpdir)/AmberTools$(VERSION)
AMBERHOME = $(SOURCE)/amber$(VERSION)_src
TOOL_SRC = $(AMBERHOME)/AmberTools/src
AMBERBIN_SRC = $(AMBERHOME)/bin
AMBERBIN_DST = $(app_bindir)/amber$(VERSION)/bin
AMBERDAT_SRC = $(AMBERHOME)/dat/antechamber
AMBERDAT_DST = $(app_bindir)/amber$(VERSION)/dat/antechamber

PATCHES	= $(wildcard AmberTools-*.patch)

test: $(SOURCE)

ENV_OPTS =
ifeq ($(OS),Windows)
CONFIG_ARGS = -cygwin gnu
ENV_OPTS = -u CC -u CXX FFLAGS=-fallow-argument-mismatch
PATCHES += flex-cygwin.patch
else
ifeq ($(OS),Darwin)
CONFIG_ARGS = clang
else
CONFIG_ARGS = gnu
endif
endif

ifeq ($(wildcard /usr/local/gfortran/*),)
FLIBDIR = /usr/local/lib
else
FLIBDIR = /usr/local/gfortran/lib
endif

install: $(SOURCE)
	cd $(AMBERHOME) && env $(ENV_OPTS) AMBERHOME='$(AMBERHOME)' ./configure --no-updates -noX11 -nofftw3 --skip-python $(CONFIG_ARGS)
	cd $(TOOL_SRC)/c9x-complex && env AMBERHOME='$(AMBERHOME)' $(MAKE) all
	cd $(TOOL_SRC)/cifparse && env AMBERHOME='$(AMBERHOME)' $(MAKE) install
	cd $(TOOL_SRC)/sqm && env AMBERHOME='$(AMBERHOME)' $(MAKE) FREEFORMAT_FLAG= install
	cd $(TOOL_SRC)/antechamber && env AMBERHOME='$(AMBERHOME)' $(MAKE) install

app-install: install
	-mkdir -p $(AMBERBIN_DST) $(AMBERDAT_DST)
	# Excluded files are because they reference netcdf and we do not need them
	$(RSYNC) $(AMBERBIN_SRC)/* --exclude nccopy --exclude ncdump --exclude ncgen --exclude ncgen3 --exclude to_be_dispatched $(AMBERBIN_DST)
	# Some executables live directly in bin, but many are shell scripts whose
	# executables are in the "to_be_dispatched" subdir; overwrite those...
	$(RSYNC) $(AMBERBIN_SRC)/to_be_dispatched/* --exclude nccopy --exclude ncdump --exclude ncgen --exclude ncgen3 $(AMBERBIN_DST)
	$(RSYNC) $(AMBERDAT_SRC)/* $(AMBERDAT_DST)
	#$(RSYNC) heme-iron.frcmod $(AMBERDAT_DST)/leap/parm
ifeq ($(OS),Darwin)
	# copy the shared libraries needed by the fortran executables and fix up their paths
	cd $(AMBERBIN_DST) && for i in sqm; do \
		for j in `otool -L $$i | awk '(NF > 1 && $$1 ~ /^$(subst /,\/,$(FLIBDIR))/) { print $$1 }'`; do \
			cp $$j $(app_libdir); \
			filename=`basename $$j`; \
			install_name_tool -id @executable_path/$$filename $(app_libdir)/$$filename; \
			install_name_tool -change $$j @executable_path/../../../lib/$$filename $$i; \
		done; \
	done
endif

$(SOURCE): $(DISTRIBUTION)
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(AMBERHOME) && patch -f -p0) < $$p ; \
	done
endif
# BUILD_TYPE is empty for local builds, otherwise one of daily, candidate, production
ifeq ($(BUILD_TYPE),daily)
	-cd $(AMBERHOME) && env AMBERHOME='$(AMBERHOME)' ./update_amber --update
else
	cd $(AMBERHOME) && env AMBERHOME='$(SOURCE)' ./update_amber --update
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ambertools/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/ambertools

clean:
	rm -rf $(SOURCE)
