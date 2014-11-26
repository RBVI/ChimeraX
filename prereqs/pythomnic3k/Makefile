FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 1.4.1
DISTRIBUTION = pythomnic3k-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pythomnic3k

all:

install: $(SOURCE)
	cd $(SOURCE) && cp -p lib/typecheck.py $(PYSITEDIR)

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	rm -rf $(SOURCE)
