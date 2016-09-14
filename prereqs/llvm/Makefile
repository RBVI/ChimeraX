PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VERSION = 3.9.0
DISTRIBUTION = llvm-$(VERSION).src.tar.xz
SOURCE = $(tmpdir)/llvm-$(VERSION).src
BUILD = $(tmpdir)/llvm-$(VERSION)

CMAKE_ARGS = -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$(build_prefix) \
	     -DLLVM_INCLUDE_EXAMPLES=OFF \
	     -DLLVM_INCLUDE_TESTS=OFF \
	     -DLLVM_TARGET_ARCH=x86_64 \
	     -DLLVM_ENABLE_ZLIB=OFF \
	     -DLLVM_ENABLE_ZLIB=OFF \

all:

install: $(SOURCE)
	#cd $(SOURCE) && \
	#  ./configure --prefix=$(build_prefix) \
	#	--enable-targets=x86_64 \
	#	--enable-optimized \
	#	--disable-docs \
	#	--disable-zlib \
	#	--disable-libedit \
	#	--disable-terminfo && \
	#  $(MAKE) install
	mkdir -p $(BUILD)
	cd $(BUILD) && cmake $(CMAKE_ARGS) $(SOURCE)
	$(MAKE) -C $(BUILD)
	$(MAKE) -C $(BUILD) install

app-install:;

$(SOURCE):
	$(bindir)/7za e -so $(DISTRIBUTION) | tar xf - -C $(tmpdir)

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
	fi
