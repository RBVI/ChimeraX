# Makefile for user quick start guide.
#
TOP	= ../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/quickstart

HTML_SRCS = index.html

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(HTML_SRCS) images $(DOCDIR)
