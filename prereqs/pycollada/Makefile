FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 2014-3-20
DISTRIBUTION = pycollada-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pycollada-$(VERSION)
PATCHES	= pycollada-python3.patch

all:

install: $(SOURCE)
	cd $(SOURCE) &&	python3 setup.py install

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)
ifneq (,$(PATCHES))
	for p in $(PATCHES); do \
		(cd $(SOURCE) && patch -f -p0) < $$p ; \
	done
endif

clean:
	rm -rf $(SOURCE)
