TOP= ../../..
include ../Makefile.bundle

# need bash for $'\n' to work
SHELL = /bin/bash

DATAFILES = $(wildcard src/icons/*.webp src/mousemode_icons/*.webp src/widgets/icons/*.webp)
CLEAN_ALWAYS += src/about.html src/*/__pycache__

build uv-wheel uv-install wheel install app-install:	src/about.html $(DATAFILES)

src/about.html:	src/about.html.in $(TOP)/copyright.txt
	sed -e '/===/d' \
		-e 's,https:.*html,<a href="&">&</a>.,' \
		$(TOP)/copyright.txt | \
		sed $$'/COPYRIGHT_NOTICE/{r /dev/stdin\nd\n}' \
		$< > $@ \
		|| (rm $@ ; exit 1)
