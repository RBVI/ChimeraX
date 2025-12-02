PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

#
# Note: LICENSE.md file is from FFmpeg source distribtion
# at https://github.com/FFmpeg/FFmpeg
#

FFMPEG_VERSION = 6.1.1

FFMPEG_PREBUILT=ffmpeg-${FFMPEG_VERSION}-${OSARCH}.exe


install: $(FFMPEG_PREBUILT)
	$(RSYNC) $(FFMPEG_PREBUILT) $(bindir)/ffmpeg$(PROG_EXT)

app-install: $(FFMPEG_PREBUILT)
	$(RSYNC) $(FFMPEG_PREBUILT) $(app_bindir)/ffmpeg$(PROG_EXT)

$(FFMPEG_PREBUILT):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/ffmpeg/$(FFMPEG_PREBUILT)

clean:
	rm -f $(FFMPEG_PREBUILT)
