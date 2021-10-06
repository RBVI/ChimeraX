PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using mesa builds of libOSMesa compiled by hand with prereqs/mesa.
#
# This package is not needed if the system's libOSMesa.so is linked
# against LLVM.

VERSION = 20.3.2

ifeq ($(OS),Darwin)
PREBUILT ?= 1
PLATFORM = mac
endif
ifeq ($(OS),Linux)
PLATFORM = linux
endif
ifeq ($(OS),Windows)
PLATFORM = windows
endif

DISTRIBUTION = osmesa-libs-$(VERSION)-$(PLATFORM).tar.bz2
SOURCE = $(tmpdir)/osmesa-$(VERSION)

install: $(SOURCE)
	$(RSYNC) $(SOURCE)/osmesa $(PYSITEDIR)

app-install: $(SOURCE)
	$(RSYNC) $(SOURCE)/osmesa $(APP_PYSITEDIR)

$(SOURCE): $(DISTRIBUTION)
	-mkdir $(SOURCE)
	$(RSYNC) osmesa $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)/osmesa

$(DISTRIBUTION):
ifeq (1,$(PREBUILT))
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/osmesa/$(DISTRIBUTION)
else
	(cd $(shlibdir) && tar jcf - libOSMesa.*$(SHLIB_EXT)*) > $(DISTRIBUTION) || (rm -f $(DISTRIBUTION); exit 1)
endif

upload_new_version: $(DISTRIBUTION)
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/osmesa

clean:
	rm -rf $(SOURCE)
