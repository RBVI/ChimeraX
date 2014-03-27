FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Leap library is built for Python 3.3.
VERSION = 1.0.9
DISTRIBUTION = leap-$(VERSION).tar.gz
SOURCE = $(tmpdir)/leap-$(VERSION)

all:

install: $(SOURCE)
ifeq ($(OS),Darwin)
	-mkdir $(PYSITEDIR)/LeapMotion
	$(RSYNC) $(SOURCE)/darwin/ $(PYSITEDIR)/LeapMotion
endif

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
