# Makefile for user quick start guide.
#
TOP	= ../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/user

SRCS = advantages.html index.html userdocs.css commands tools
IMAGES = ambient.png contacts2.png surf4hhb.png Chimera2-icon.svg

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

clean:
