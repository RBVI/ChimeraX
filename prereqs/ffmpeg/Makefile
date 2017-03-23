PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

FFMPEG_VERSION = 2.3.2
FFMPEG_DISTRIB = ffmpeg-$(FFMPEG_VERSION).tar.bz2
FFMPEG_SOURCE = $(tmpdir)/ffmpeg-$(FFMPEG_VERSION)
FFMPEG_EXE = $(FFMPEG_SOURCE)/ffmpeg

# Windows ffmpeg is precompiled statically linked executable from Zeranoe.
FFMPEG_WIN_EXE = ffmpeg-3.2.4-win64.exe

LIBX264_VERSION = snapshot-20140810-2245-stable
LIBX264_DISTRIB = x264-$(LIBX264_VERSION).tar.bz2
LIBX264_SOURCE = $(tmpdir)/x264-$(LIBX264_VERSION)
LIBX264 = $(libdir)/libx264.a

YASM_VERSION = 1.3.0
YASM_DISTRIB = yasm-$(YASM_VERSION).tar.bz2
YASM_SOURCE = $(tmpdir)/yasm-$(YASM_VERSION)
YASM_EXE = $(bindir)/yasm

FFMPEG_OPTS	= --disable-network \
			--disable-doc \
			--disable-devices \
			--disable-ffplay \
			--disable-ffserver \
			--disable-ffprobe \
			--disable-mmx \
			--enable-gpl \
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

else

all: $(FFMPEG_EXE)

install: $(FFMPEG_EXE)
	$(RSYNC) $(FFMPEG_EXE) $(bindir)

app-install:
	$(RSYNC) $(FFMPEG_EXE) $(app_bindir)

clean:
	rm -rf $(FFMPEG_SOURCE) $(YASM_SOURCE) $(LIBX264_SOURCE)

$(FFMPEG_EXE): $(FFMPEG_SOURCE) video_codecs
	cd $(FFMPEG_SOURCE); ./configure $(FFMPEG_OPTS) ; $(MAKE)

$(FFMPEG_SOURCE):
	tar jxf $(FFMPEG_DISTRIB) -C $(tmpdir)

video_codecs: $(LIBX264)

$(YASM_EXE): $(YASM_SOURCE)
	cd $(YASM_SOURCE) ; ./configure --prefix=$(build_prefix); $(MAKE) install

$(YASM_SOURCE): $(YASM_DISTRIB)
	tar jxf $(YASM_DISTRIB) -C $(tmpdir)

$(LIBX264): $(LIBX264_SOURCE) $(YASM_EXE)
	cd $(LIBX264_SOURCE) ; \
	env PATH=$(bindir):$(PATH) ./configure --prefix=$(build_prefix) --enable-pic; \
	env PATH=$(bindir):$(PATH) $(MAKE) lib-static install install-lib-static install-lib-dev

$(LIBX264_SOURCE):
	tar jxf $(LIBX264_DISTRIB) -C $(tmpdir)

endif
