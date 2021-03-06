PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 9.20.1
DISTRIBUTION = p7zip_$(VERSION)_src_all.tar.bz2
SOURCE = $(tmpdir)/p7zip_$(VERSION)

PATCHES	= $(wildcard p7zip-$(VERSION)-*.patch)

ifeq ($(OSARCH),Linux)
	MACHINE = linux_any_cpu
else ifeq ($(OSARCH),Linux64)
	MACHINE = linux_amd64_asm
else ifeq ($(OSARCH),DarwinIntel)
	MACHINE = macosx_32bits_asm
else ifeq ($(OSARCH),DarwinIntel64)
	MACHINE = macosx_64bits
else
	$(error Unsupported $(OSARCH))
endif

all:

install: $(SOURCE)
	cd $(SOURCE) && cp makefile.$(MACHINE) makefile.machine
	cd $(SOURCE) && $(MAKE) all && $(MAKE) DEST_HOME=$(build_prefix) install
	# allow rm -r to work on build tree
	chmod -R u+w $(datadir)/doc/p7zip/DOCS

app-install:;

$(SOURCE): $(DISTRIBUTION)
	tar jxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/p7zip/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/p7zip

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
