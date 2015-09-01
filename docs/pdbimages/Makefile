# Makefile for user quick start guide.
#
TOP	= ../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/pdbimages

SRCS = index.html tiled.jpg

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(DOCDIR)
