TOP = ..

include $(TOP)/mk/config.make

ifdef WIN32
RST2HTML = $(shell cygpath -m $(bindir)/Scripts/rst2html.py)
else
RST2HTML = $(bindir)/rst2html.py
endif

SUBDIRS = devel pdbimages quickstart user download

DOCDIR = $(app_datadir)/docs

SRCS = $(wildcard *.html)
IMAGES = $(wildcard *.png)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

install: license.html
	if [ ! -d "$(DOCDIR)" ]; then mkdir -p $(DOCDIR); fi
	$(RSYNC) $(SRCS) $(IMAGES) $(DOCDIR)

license.html: license.rst
	$(PYTHON_EXE) $(RST2HTML) \
		--smart-quotes=alt \
		--stylesheet-path=license.css \
		license.rst > license.html

clean:
	rm -f license.html
