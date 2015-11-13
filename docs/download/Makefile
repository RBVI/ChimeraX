# Makefile for download page.
#
TOP	= ../..
include $(TOP)/mk/config.make

DOCDIR = $(app_datadir)/docs/download

HTML_SRCS = chimerax.html

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(HTML_SRCS) $(DOCDIR)

clean:
