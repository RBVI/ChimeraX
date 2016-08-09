TOP = ..

include $(TOP)/mk/config.make

SUBDIRS = devel pdbimages quickstart user download

DOCDIR = $(app_datadir)/docs

SRCS = $(wildcard *.html)
IMAGES = $(wildcard *.png)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install:
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)
