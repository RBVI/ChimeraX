FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 0.28
DISTRIBUTION = pkg-config-$(VERSION).tar.gz
SOURCE = $(tmpdir)/pkg-config-$(VERSION)

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  ./configure --prefix=$(build_prefix) --with-internal-glib && \
	  make install

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
