# Makefile for user quick start guide.
#
TOP	= ../..
include $(TOP)/mk/config.make

SUBDIRS = tools

DOCDIR = $(app_datadir)/docs/user

SRCS = advantages.html index.html userdocs.css commands tools
IMAGES = ambient.png blimp.png contacts2.png ChimeraX-icon.svg

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

clean:
