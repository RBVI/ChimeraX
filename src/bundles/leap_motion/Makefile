include ../Makefile.bundle

TOP	= ../../..
include $(TOP)/mk/config.make

LEAP_LIBRARY = leapmotion-4.0.tar.gz
DISTCLEAN = $(LEAP_LIBRARY)

wheel install app-install: $(LEAP_LIBRARY)

$(LEAP_LIBRARY):
	$(call FETCH_PREREQ_TO,$@,$(PREREQS_ARCHIVE)/leapmotion/$(LEAP_LIBRARY))
	tar xf $(LEAP_LIBRARY) --no-same-owner
