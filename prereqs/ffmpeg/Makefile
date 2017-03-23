PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

FFMPEG_VERSION = 3.2.4
FFMPEG_DISTRIB = ffmpeg-$(FFMPEG_VERSION).tar.bz2
FFMPEG_SOURCE = $(tmpdir)/ffmpeg-$(FFMPEG_VERSION)
FFMPEG_EXE = $(FFMPEG_SOURCE)/ffmpeg

# Windows ffmpeg is precompiled statically linked executable from Zeranoe.
FFMPEG_WIN_EXE = ffmpeg-3.2.4-win64.exe

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

ifeq ($(OS),Windows)

all:

install:
	$(RSYNC) $(FFMPEG_WIN_EXE) $(bindir)/ffmpeg$(PROG_EXT)

app-install:
	$(RSYNC) $(FFMPEG_WIN_EXE) $(app_bindir)/ffmpeg$(PROG_EXT)

clean:

else

all: $(FFMPEG_EXE)

install: $(FFMPEG_EXE)
	$(MAKE) -C $(FFMPEG_SOURCE) install

app-install:
	$(RSYNC) $(bindir)/ffmpeg$(PROG_EXT) $(app_bindir)
	$(RSYNC) --exclude examples $(datadir)/ffmpeg $(app_datadir)/ffmpeg

clean:
	rm -rf $(FFMPEG_SOURCE) $(YASM_SOURCE) $(LIBX264_SOURCE) \
		$(LIBOGG_SOURCE) $(LIBTHEORA_SOURCE) $(LIBVPX_SOURCE)

$(FFMPEG_EXE): $(FFMPEG_SOURCE) video_codecs
	cd $(FFMPEG_SOURCE); ./configure $(FFMPEG_OPTS) ; $(MAKE)

$(FFMPEG_SOURCE):
	tar jxf $(FFMPEG_DISTRIB) -C $(tmpdir)

video_codecs: $(LIBTHEORA) $(LIBVPX) $(LIBX264)

yasm $(YASM_EXE): $(YASM_SOURCE)
	cd $(YASM_SOURCE) ; ./configure --prefix=$(build_prefix); $(MAKE) install

$(YASM_SOURCE): $(YASM_DISTRIB)
	tar jxf $(YASM_DISTRIB) -C $(tmpdir)

libx264 $(LIBX264): $(LIBX264_SOURCE) $(YASM_EXE)
	cd $(LIBX264_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --enable-pic; \
	env PATH=$(bindir):$(PATH) $(MAKE) lib-static install install-lib-static install-lib-dev

$(LIBX264_SOURCE):
	$(bindir)/7za e -so $(LIBX264_DISTRIB) | tar xf - -C $(tmpdir)

libogg $(LIBOGG): $(LIBOGG_SOURCE)
	cd $(LIBOGG_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --disable-shared; \
	env PATH=$(bindir):$(PATH) $(MAKE) install

$(LIBOGG_SOURCE):
	$(bindir)/7za e -so $(LIBOGG_DISTRIB) | tar xf - -C $(tmpdir)

libtheora $(LIBTHEORA): $(LIBTHEORA_SOURCE) $(LIBOGG)
	-cd $(LIBTHEORA_SOURCE) ; ./autogen.sh
	cd $(LIBTHEORA_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --disable-shared --with-ogg=$(LIBOGG) --disable-oggtest --disable-vorbistest --disable-sdltest --disable-examples; \
	env PATH=$(bindir):$(PATH) $(MAKE) install

$(LIBTHEORA_SOURCE):
	$(bindir)/7za e -so $(LIBTHEORA_DISTRIB) | tar xf - -C $(tmpdir)

libvpx $(LIBVPX): $(LIBVPX_SOURCE)
	cd $(LIBVPX_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --enable-pic --disable-examples --disable-tools --disable-docs --enable-vp8 --enable-vp9 --enable-runtime-cpu-detect --disable-shared --disable-unit-tests; \
	env PATH=$(bindir):$(PATH) $(MAKE) install

$(LIBVPX_SOURCE):
	$(bindir)/7za e -so $(LIBVPX_DISTRIB) | tar xf - -C $(tmpdir)

endif
