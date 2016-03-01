# Makefile for user guide
#
TOP	= ../..
include $(TOP)/mk/config.make

SUBDIRS = tools

DOCDIR = $(app_datadir)/docs/user

SRCS = $(wildcard *.html) userdocs.css commands
IMAGES = $(wildcard *.png) ChimeraX-icon.svg

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

clean:
