TOP = ../../..
include $(TOP)/mk/config.make

include ../Makefile.bundle

HOLOPLAYCORE_LIBRARY = holoplaycore-0.1.0.tar.gz
CLEAN_ALWAYS = src/lib
DISTCLEAN = $(HOLOPLAYCORE_LIBRARY)

wheel install app-install: $(HOLOPLAYCORE_LIBRARY)

$(HOLOPLAYCORE_LIBRARY):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/lookingglass/$(HOLOPLAYCORE_LIBRARY)
	tar xf $(HOLOPLAYCORE_LIBRARY)
