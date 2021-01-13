PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Using mesa builds of libOSMesa compiled by hand with prereqs/mesa.

VERSION = 20.3.2

ifeq ($(OS),Darwin)
PLATFORM = mac
endif
ifeq ($(OS),Linux)
PLATFORM = linux
endif
ifeq ($(OS),Windows)
PLATFORM = windows
endif

DISTRIBUTION = osmesa-$(VERSION)-$(PLATFORM).tar.bz2
SOURCE = $(tmpdir)/osmesa-$(VERSION)

install: $(SOURCE)
	$(RSYNC) $(SOURCE)/osmesa $(PYSITEDIR)

app-install: $(SOURCE)
	$(RSYNC) $(SOURCE)/osmesa $(APP_PYSITEDIR)

$(SOURCE): $(DISTRIBUTION)
	-mkdir $(SOURCE)
	tar xf $(DISTRIBUTION) -C $(SOURCE)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/osmesa/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/osmesa

clean:
	rm -rf $(SOURCE)
