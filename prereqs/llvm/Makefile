PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Requires cmake 3.4.3 or newer

VERSION = 3.9.0
DISTRIBUTION = llvm-$(VERSION).src.tar.xz
SOURCE = $(tmpdir)/llvm-$(VERSION).src
BUILD = $(tmpdir)/llvm-$(VERSION)

CMAKE_ARGS = -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$(build_prefix) \
	     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	     -DLLVM_TARGETS_TO_BUILD=X86 \
	     -DLLVM_INCLUDE_TOOLS=OFF \
	     -DLLVM_INCLUDE_TESTS=OFF \
	     -DLLVM_INCLUDE_EXAMPLES=OFF \
	     -DLLVM_INCLUDE_UTILS=ON \
	     -DLLVM_INCLUDE_DOCS=OFF \
	     -DLLVM_INCLUDE_BENCHMARKS=OFF \
	     -DLLVM_BUILD_UTILS=OFF \
	     -DLLVM_BUILD_TOOLS=OFF \
	     -DLLVM_BUILD_LLVM_DYLIB=ON \
	     -DLLVM_ENABLE_LIBCXX=OFF \
	     -DLLVM_ENABLE_FFI=OFF \
	     -DLLVM_ENABLE_LIBCXX=OFF

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

$(SOURCE): $(DISTRIBUTION)
	tar -xf $(DISTRIBUTION) -C $(tmpdir)

$(DISTRIBUTION):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/llvm/$(DISTRIBUTION)

upload_new_version:
	$(RSYNC) $(DISTRIBUTION) $(PREREQS_UPLOAD)/llvm

clean:
	if [ -d $(SOURCE) ] ; then \
		chmod -R +wX $(SOURCE) ; \
		rm -rf $(SOURCE) ; \
		rm -rf $(BUILD) ; \
	fi
