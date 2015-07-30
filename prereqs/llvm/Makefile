FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.6.2
DISTRIBUTION = llvm-$(VERSION).src.tar.gz
SOURCE = $(tmpdir)/llvm-$(VERSION).src

all:

install: $(SOURCE)
	cd $(SOURCE) && \
	  ./configure --prefix=$(build_prefix) \
		--enable-targets=x86_64 \
		--enable-optimized \
		--disable-docs \
		--disable-zlib \
		--disable-libedit \
		--disable-terminfo && \
	  make install

app-install:;

$(SOURCE):
	tar zxf $(DISTRIBUTION) -C $(tmpdir)

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
