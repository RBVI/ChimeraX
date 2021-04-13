TOP= ../../..
include ../Makefile.bundle

# need bash for $'\n' to work
SHELL = /bin/bash

DATAFILES = $(wildcard src/icons/*.png src/mousemode_icons/*.png src/widgets/icons/*.png)
CLEAN_ALWAYS += src/about.html src/*/__pycache__

build wheel install app-install:	src/about.html $(DATAFILES)

src/about.html:	src/about.html.in
	sed -e '/===/d' \
		-e 's,https:.*html,<a href="&">&</a>.,' \
		$(TOP)/copyright.txt | \
		sed $$'/COPYRIGHT_NOTICE/{r /dev/stdin\nd\n}' \
		$< > $@ \
		|| (rm $@ ; exit 1)
