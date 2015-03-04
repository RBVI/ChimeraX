FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.4.4_beta
DISTRIBUTION = ovr_sdk_macos_$(VERSION).tar.gz
SOURCE = $(tmpdir)/OculusSDK

all:

install: $(SOURCE)
ifeq ($(OS),Darwin)
	$(RSYNC) $(SOURCE)/LibOVR $(libdir)
endif

app-install:


$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
