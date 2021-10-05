PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

FFMPEG_VERSION = 3.2.4

ifneq ($(OS),Linux)
USE_PREBUILT = 1
endif

FFMPEG_PREBUILT = ffmpeg-$(FFMPEG_VERSION)-$(OS).exe

# The following variables are from compiling from source code.
# We no longer do that unless we want to update the source libraries.

ifeq ($(OS),Windows)
# Windows zip file is needed on all platforms because it has all of the
# sublicences in it that are needed for the documentation.
FFMPEG_DISTRIB = ffmpeg-$(FFMPEG_VERSION)-win64-static.zip
FFMPEG_SOURCE = $(tmpdir)/ffmpeg-$(FFMPEG_VERSION)-win64-static
FFMPEG_EXE = $(FFMPEG_SOURCE)/bin/ffmpeg.exe
else
FFMPEG_DISTRIB = ffmpeg-$(FFMPEG_VERSION).tar.bz2
FFMPEG_SOURCE = $(tmpdir)/ffmpeg-$(FFMPEG_VERSION)
FFMPEG_EXE = $(FFMPEG_SOURCE)/ffmpeg

LIBX264_VERSION = 20170123-90a61ec
LIBX264_DISTRIB = x264-$(LIBX264_VERSION).tar.xz
LIBX264_SOURCE = $(tmpdir)/x264-$(LIBX264_VERSION)
LIBX264 = $(libdir)/libx264.a

LIBOGG_VERSION = 1.3.2
LIBOGG_DISTRIB = libogg-$(LIBOGG_VERSION).tar.xz
LIBOGG_SOURCE = $(tmpdir)/libogg-$(LIBOGG_VERSION)
LIBOGG = $(libdir)/libogg.a

LIBTHEORA_VERSION = 1.1.1
LIBTHEORA_DISTRIB = libtheora-$(LIBTHEORA_VERSION).tar.xz
LIBTHEORA_SOURCE = $(tmpdir)/libtheora-$(LIBTHEORA_VERSION)
LIBTHEORA = $(libdir)/libtheora.a

LIBVPX_VERSION = 1.6.1
LIBVPX_DISTRIB = libvpx-$(LIBVPX_VERSION).tar.xz
LIBVPX_SOURCE = $(tmpdir)/libvpx-$(LIBVPX_VERSION)
LIBVPX = $(libdir)/libvpx.a

YASM_VERSION = 1.3.0
YASM_DISTRIB = yasm-$(YASM_VERSION).tar.bz2
YASM_SOURCE = $(tmpdir)/yasm-$(YASM_VERSION)
YASM_EXE = $(bindir)/yasm

FFMPEG_OPTS	=  --prefix=$(build_prefix) \
		   	--enable-gpl --enable-version3 \
			--disable-network \
			--disable-doc \
			--disable-devices \
			--disable-ffplay \
			--disable-ffserver \
			--disable-ffprobe \
			--enable-libtheora \
			--enable-libvpx \
			--enable-libx264 \
			--yasmexe=$(bindir)/yasm \
			--optflags=-O \
			--extra-cflags="-I$(includedir)" \
			--extra-ldflags="$(LDFLAGS) -L$(libdir)" \
			--extra-libs="-ldl"
endif


ifdef USE_PREBUILT
install: $(FFMPEG_PREBUILT)
	$(RSYNC) $(FFMPEG_PREBUILT) $(bindir)/ffmpeg$(PROG_EXT)

app-install: $(FFMPEG_PREBUILT)
	$(RSYNC) $(FFMPEG_PREBUILT) $(app_bindir)/ffmpeg$(PROG_EXT)

$(FFMPEG_PREBUILT):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(FFMPEG_PREBUILT)
	chmod a+x $(FFMPEG_PREBUILT)

upload_new_version:
	$(RSYNC) $(FFMPEG_PREBUILT) $(PREREQS_UPLOAD)/ffmpeg

else
install: $(FFMPEG_EXE)
	$(RSYNC) $(FFMPEG_EXE) $(bindir)/ffmpeg$(PROG_EXT)

app-install: $(FFMPEG_EXE)
	$(RSYNC) $(FFMPEG_EXE) $(app_bindir)/ffmpeg$(PROG_EXT)
endif

ifeq ($(OS),Windows)

from-binary-distribution: $(FFMPEG_EXE)
	$(RSYNC) $(FFMPEG_EXE) $(FFMPEG_PREBUILT)

$(FFMPEG_EXE): $(FFMPEG_DISTRIB)
	unzip $(FFMPEG_DISTRIB) -d $(tmpdir)
	touch $(FFMPEG_EXE)

$(FFMPEG_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(FFMPEG_DISTRIB)

clean:
	rm -rf $(FFMPEG_PREBUILT) $(FFMPEG_DISTRIB)

else

compile: $(FFMPEG_EXE)
	$(MAKE) -C $(FFMPEG_SOURCE) install
	$(RSYNC) $(bindir)/ffmpeg$(PROG_EXT) $(FFMPEG_PREBUILT)

clean:
	rm -rf $(FFMPEG_PREBUILT) \
		$(FFMPEG_SOURCE) $(YASM_SOURCE) $(LIBX264_SOURCE) \
		$(LIBOGG_SOURCE) $(LIBTHEORA_SOURCE) $(LIBVPX_SOURCE) \
		$(FFMPEG_DISTRIB) $(YASM_DISTRIB) $(LIBX264_DISTRIB) \
		$(LIBOGG_DISTRIB) $(LIBTHEORA_DISTRIB) $(LIBVPX_DISTRIB)

$(FFMPEG_EXE): $(FFMPEG_SOURCE) video_codecs
	cd $(FFMPEG_SOURCE); ./configure $(FFMPEG_OPTS) ; $(MAKE)

$(FFMPEG_SOURCE): $(FFMPEG_DISTRIB)
	tar jxf $(FFMPEG_DISTRIB) -C $(tmpdir)

$(FFMPEG_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(FFMPEG_DISTRIB)

video_codecs: $(LIBTHEORA) $(LIBVPX) $(LIBX264)

yasm $(YASM_EXE): $(YASM_SOURCE)
	cd $(YASM_SOURCE) ; ./configure --prefix=$(build_prefix); $(MAKE) install

$(YASM_SOURCE): $(YASM_DISTRIB)
	tar -xf $(YASM_DISTRIB) -C $(tmpdir)

$(YASM_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(YASM_DISTRIB)

libx264 $(LIBX264): $(LIBX264_SOURCE) $(YASM_EXE)
	cd $(LIBX264_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --enable-pic; \
	env PATH=$(bindir):$(PATH) $(MAKE) lib-static install install-lib-static install-lib-dev

$(LIBX264_SOURCE): $(LIBX264_DISTRIB)
	tar -xf $(LIBX264_DISTRIB) -C $(tmpdir)

$(LIBX264_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(LIBX264_DISTRIB)

libogg $(LIBOGG): $(LIBOGG_SOURCE)
	cd $(LIBOGG_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --disable-shared; \
	env PATH=$(bindir):$(PATH) $(MAKE) install

$(LIBOGG_SOURCE): $(LIBOGG_DISTRIB)
	tar -xf $(LIBOGG_DISTRIB) -C $(tmpdir)

$(LIBOGG_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(LIBOGG_DISTRIB)

libtheora $(LIBTHEORA): $(LIBTHEORA_SOURCE) $(LIBOGG)
ifeq ($(OS),Darwin)
	ln -sfF $(datadir)/libtool/build-aux/ltmain.sh $(LIBTHEORA_SOURCE)/ltmain.sh
else ifneq (,$(wildcard /usr/share/libtool/build-aux/ltmain.sh))
	-cd $(LIBTHEORA_SOURCE) ; ./autogen.sh
endif
	cd $(LIBTHEORA_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --disable-shared --with-ogg=$(LIBOGG) --disable-oggtest --disable-vorbistest --disable-sdltest --disable-examples --disable-spec
ifeq ($(OS),Darwin)
	# somehow the wrong libtool ends up in source tree, so overwrite it
	ln -sfF $(bindir)/libtool $(LIBTHEORA_SOURCE)/libtool
endif
	env PATH=$(bindir):$(PATH) $(MAKE) -C $(LIBTHEORA_SOURCE) install
ifeq ($(OS),Darwin)
	rm -f $(libdir)/libtheora*.dylib
endif

$(LIBTHEORA_SOURCE): $(LIBTHEORA_DISTRIB)
	tar -xf $(LIBTHEORA_DISTRIB) -C $(tmpdir)

$(LIBTHEORA_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(LIBTHEORA_DISTRIB)

libvpx $(LIBVPX): $(LIBVPX_SOURCE) $(YASM_EXE)
	cd $(LIBVPX_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --enable-pic --disable-examples --disable-tools --disable-docs --enable-vp8 --enable-vp9 --enable-runtime-cpu-detect --disable-shared --disable-unit-tests; \
	env PATH=$(bindir):$(PATH) $(MAKE) install

$(LIBVPX_SOURCE): $(LIBVPX_DISTRIB)
	tar -xf $(LIBVPX_DISTRIB) -C $(tmpdir)

$(LIBVPX_DISTRIB):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(LIBVPX_DISTRIB)

upload_new_sources:
	$(RSYNC) $(FFMPEG_DISTRIB) $(YASM_DISTRIB) $(LIBX264_DISTRIB) $(LIBOGG_DISTRIB) $(LIBTHEORA_DISTRIB) $(LIBVPX_DISTRIB) $(PREREQS_UPLOAD)/ffmpeg

endif
